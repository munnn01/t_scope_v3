"""Strict, independently seeded T-SCOPE v3 training and paired evaluation.

No v2 checkpoint, synthetic data, or lightweight detector fallback is accepted
by an experiment. Unit tests can exercise batch_loss with small local components.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from scripts.experiment_protocol import (
    CODEC_CONFIGURATION, CRFS, EXPECTED_TEST_FRAME_IDS, METRIC_POLICY,
    TRAIN_SEQUENCES, VAL_SEQUENCES, sha256_file,
)
from scripts.v3_protocol import VARIANTS, build_model, variant_settings
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from video_codecs.hybrid_x265_ste import HybridX265STE


DEFAULTS = {
    "epochs": 36, "min_epochs": 12, "patience": 10, "min_delta": 0.0005,
    "batch_size": 2, "grad_accum_steps": 2, "img_size": 512, "clip_len": 4,
    "lr": 0.00015, "num_workers": 2, "max_hours": 7.8, "val_subsample": 2,
}


def current_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def make_job(variant, seed, **overrides):
    unknown = set(overrides) - set(DEFAULTS) - {"source_revision", "run_id"}
    if unknown:
        raise ValueError(f"Unknown job options: {sorted(unknown)}")
    settings = variant_settings(variant)
    job = {"schema_version": 3, "job_id": f"{variant}_seed{seed}",
           "variant": variant, "seed": seed, "settings": settings,
           **DEFAULTS, **overrides,
           "train_sequences": list(TRAIN_SEQUENCES), "val_sequences": list(VAL_SEQUENCES),
           "teacher": "fasterrcnn_resnet50_fpn_v2", "train_yolo": False,
           "codec": copy.deepcopy(CODEC_CONFIGURATION),
           "source_revision": overrides.get("source_revision") or current_revision()}
    validate_job(job)
    return job


def validate_job(job):
    if job.get("schema_version") != 3 or job.get("variant") not in VARIANTS:
        raise ValueError("Expected a schema-3 job with a known variant")
    if type(job.get("seed")) is not int or not 0 <= job["seed"] < 2 ** 32:
        raise ValueError("Seed must be a uint32 integer")
    if job.get("job_id") != f"{job['variant']}_seed{job['seed']}":
        raise ValueError("Job identity differs from its variant/seed")
    if job.get("settings") != variant_settings(job["variant"]):
        raise ValueError("Job settings differ from its versioned variant")
    for key in ("epochs", "min_epochs", "patience", "batch_size", "grad_accum_steps", "val_subsample"):
        if type(job.get(key)) is not int or job[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(job.get("num_workers")) is not int or job["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if job["min_epochs"] > job["epochs"]:
        raise ValueError("min_epochs exceeds epochs")
    if type(job.get("img_size")) is not int or job["img_size"] < 32 or job["img_size"] % 16:
        raise ValueError("img_size must be a multiple of 16, at least 32")
    if type(job.get("clip_len")) is not int or job["clip_len"] != 4:
        raise ValueError("The initial paired v3 protocol requires clip_len=4")
    for key in ("lr", "max_hours", "min_delta"):
        if type(job.get(key)) not in (float, int) or not math.isfinite(job[key]) or job[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if job.get("train_sequences") != TRAIN_SEQUENCES or job.get("val_sequences") != VAL_SEQUENCES:
        raise ValueError("Training/validation splits differ from the v3 protocol")
    if job.get("teacher") != "fasterrcnn_resnet50_fpn_v2" or job.get("train_yolo") is not False:
        raise ValueError("v3 training uses pretrained Faster R-CNN only; YOLO is held out")
    if job.get("codec") != CODEC_CONFIGURATION:
        raise ValueError("Codec configuration differs from the matched protocol")
    revision = job.get("source_revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("A full pinned source_revision is required")
    return job


def load_real_dataset(job, root, split):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown data split")
    dataset = RealVCMVideoDataset(
        root, shard_id=0, num_shards=1, clip_len=job["clip_len"],
        stride=2 if split == "train" else job["clip_len"], img_size=job["img_size"],
        is_train=split == "train", enforce_disjoint_split=True, split=split,
    )
    allowed = set(TRAIN_SEQUENCES if split == "train" else VAL_SEQUENCES if split == "val" else ["MOT17-11", "MOT17-13"])
    actual = {meta["seq_name"].split("-FRCNN")[0] for meta in dataset.clips}
    if not dataset.clips or actual != allowed or any(meta["type"] != "mot_frames" for meta in dataset.clips):
        raise ValueError(f"{split} requires all real MOT17 sequences {sorted(allowed)}; no synthetic fallback")
    if split == "test" and dataset.physical_frame_manifest() != EXPECTED_TEST_FRAME_IDS:
        raise ValueError("Held-out physical frame coverage differs from the paired 1,648-frame protocol")
    return dataset


class EpochDataset(Dataset):
    """Make crops a function of seed/epoch/clip, independent of model RNG or workers."""
    def __init__(self, dataset, seed, epoch):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((self.seed + 1000003 * self.epoch + 9176 * index) % (2 ** 63 - 1))
            return self.dataset[index]


def require_teacher(teacher, training=False, require_yolo=False):
    if not teacher.use_torchvision or teacher.detector is None:
        raise RuntimeError("Experiments require actual pretrained Faster R-CNN weights")
    if training and teacher.use_yolov8s:
        raise RuntimeError("YOLO must remain unseen during v3 training")
    if require_yolo and not teacher.use_yolov8s:
        raise RuntimeError("Requested evaluation-only YOLO weights did not load")


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_resume(checkpoint, job):
    if checkpoint.get("schema_version") != 3 or checkpoint.get("job") != job:
        raise ValueError("Cannot resume another job, revision, configuration or legacy checkpoint")
    for key in ("model_state", "optimizer_state", "scaler_state", "epoch", "next_batch", "history"):
        if key not in checkpoint:
            raise ValueError(f"Incomplete resume checkpoint: {key}")


def restore_checkpoint(path, job, model, optimizer, scaler):
    """Read through the safe loader and restore every trainable execution state."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate_resume(state, job)
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scaler.load_state_dict(state["scaler_state"])
    return state


def make_criterion(job):
    settings = job["settings"]
    temporal = not settings["disable_temporal"]
    return VCMTemporalPreEditLoss(
        lambda_temp_bg=settings["lambda_temp_bg"], lambda_temp_obj=settings["lambda_temp_obj"],
        lambda_dqp_mono=0.4 if temporal else 0.0,
        lambda_dqp_smooth=0.15 if temporal else 0.0,
    )


@torch.no_grad()
def validation_loss(model, codec, criterion, teacher, dataset, job, device, deadline):
    was_training = model.training
    model.eval()
    loader = DataLoader(Subset(dataset, list(range(0, len(dataset), job["val_subsample"]))),
                        batch_size=job["batch_size"], shuffle=False, num_workers=0)
    total, count = 0.0, 0
    try:
        for batch in loader:
            for severity in (0.3, 0.7):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Validation session deadline exhausted")
                sev = torch.full((batch["video"].shape[0],), severity, device=device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    value = batch_loss(model, codec, criterion, teacher, batch, job, device, sev)["loss_total"]
                if not torch.isfinite(value):
                    raise RuntimeError("Nonfinite validation loss")
                total += value.item()
                count += 1
        if not count:
            raise ValueError("Validation requires a nonempty dataset")
        return total / count
    finally:
        model.train(was_training)


def batch_loss(model, codec, criterion, teacher, batch, job, device, severity=None):
    video = batch["video"].to(device)
    b, t, _, h, w = video.shape
    if severity is None:
        severity = torch.rand(b, device=device)
    out = model(video, torch.zeros(b, dtype=torch.long, device=device), severity,
                reuse_iframe_qp=False, disable_temporal=job["settings"]["disable_temporal"])
    decoded = codec(out["i_edit"], severity, out["ctu_qp_offsets"])
    with torch.no_grad():
        teacher_features = teacher.extract_fpn_features(video.reshape(b * t, 3, h, w))
    features = teacher.extract_fpn_features(decoded["video_recon"].reshape(b * t, 3, h, w))
    return criterion(
        i_edit=out["i_edit"], i_recon=decoded["video_recon"], attn_map=out["attn_map"],
        delta_obj=out["delta_obj"], delta_bg=out["delta_bg"],
        rate_proxy_bpp=decoded["rate_proxy_bpp"], m_attn=batch["m_attn"].to(device),
        m_det=batch["m_det"].to(device), teacher_fpn_feats=teacher_features,
        student_fpn_feats=features, i_frame_bpp_proxy=decoded.get("i_frame_bpp_proxy"),
        p_frame_bpp_proxy=decoded.get("p_frame_bpp_proxy"), ctu_qp_offsets=out["ctu_qp_offsets"],
        i_orig=video, aligned_prev_edit=out.get("aligned_prev_edit"),
        aligned_prev_orig=out.get("aligned_prev_orig"), warp_valid_mask=out.get("warp_valid_mask"),
    )


def data_fingerprint(dataset):
    """Hash unique physical images and GT without retaining machine-specific paths."""
    images, annotations = {}, {}
    for meta in dataset.clips:
        sequence = meta["seq_name"].split("-FRCNN")[0]
        for path in meta["frame_paths"]:
            images[f"{sequence}/{path.name}"] = path
        annotations[sequence] = meta["frame_paths"][0].parent.parent / "gt" / "gt.txt"
    digest = hashlib.sha256()
    for key, path in sorted(images.items()):
        digest.update(f"{key}:{sha256_file(path)}\n".encode())
    return {"physical_frames": len(images), "images_sha256": digest.hexdigest(),
            "gt_sha256": {key: sha256_file(path) for key, path in sorted(annotations.items())}}


def execution_provenance():
    import torchvision
    files = ["models/temporal_scope.py", "models/scope_reference.py", "models/frozen_detector_fpn.py",
             "scripts/run_v3.py", "scripts/v3_protocol.py", "losses/vcm_temporal_loss.py",
             "vcm_datasets/vcm_video_datasets.py", "video_codecs/hybrid_x265_ste.py"]
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "source_sha256": {name: sha256_file(ROOT / name) for name in files},
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def train(job, data_root, output_dir, resume=None, canary=False):
    validate_job(job)
    if current_revision() != job["source_revision"]:
        raise RuntimeError("Local source does not match the pinned job revision")
    dataset = load_real_dataset(job, data_root, "train")
    val_dataset = load_real_dataset(job, data_root, "val")
    settings = job["settings"]
    import shutil
    if (canary or settings["use_real_x265_ste"]) and (not shutil.which("ffmpeg") or not shutil.which("ffprobe")):
        raise RuntimeError("Real-codec training/canary requires ffmpeg and ffprobe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if canary and device.type != "cuda":
        raise RuntimeError("Production canary requires a CUDA GPU")
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    model = build_model(job["variant"]).to(device)
    teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=False).to(device).eval()
    require_teacher(teacher, training=True)
    codec = HybridX265STE(use_real_x265_in_forward=canary or settings["use_real_x265_ste"]).to(device)
    criterion = make_criterion(job).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / (job["job_id"] + (".canary.pt" if canary else ".last.pt"))
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("Run output already exists; choose another directory or explicitly resume")
    provenance = execution_provenance()
    provenance["training_data"] = data_fingerprint(dataset)
    provenance["validation_data"] = data_fingerprint(val_dataset)
    if settings["use_real_x265_ste"] or canary:
        provenance["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    history, best_state, best_loss, best_epoch = [], None, float("inf"), 0
    start_epoch, skip_batches, updates = 1, 0, 0
    if resume:
        state = restore_checkpoint(resume, job, model, optimizer, scaler)
        if state["provenance"]["training_data"] != provenance["training_data"] or state["provenance"]["validation_data"] != provenance["validation_data"]:
            raise ValueError("Resume data fingerprints differ")
        if state.get("status") == "canary_passed":
            raise ValueError("Canary output is not a production-training resume point")
        if state.get("status") in ("completed", "converged"):
            return {"status": state["status"], "checkpoint": str(resume)}
        history, best_state, best_epoch = state["history"], state["best_state"], state["best_epoch"]
        best_loss = float("inf") if state["best_loss"] is None else state["best_loss"]
        start_epoch, skip_batches, updates = state["epoch"], state["next_batch"], state["optimizer_updates"]
    started = time.monotonic()
    status, next_epoch, next_batch, phase = "running", start_epoch, skip_batches, "train"
    optimizer.zero_grad(set_to_none=True)

    def snapshot():
        save_checkpoint(checkpoint_path, {
            "schema_version": 3, "job": job, "status": status,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "best_state": best_state,
            "best_loss": best_loss if math.isfinite(best_loss) else None,
            "best_epoch": best_epoch, "epoch": next_epoch, "next_batch": next_batch,
            "optimizer_updates": updates, "history": history, "provenance": provenance,
            "phase": phase,
            "elapsed_session_seconds": time.monotonic() - started,
            "final_ready": False,
        })

    for epoch in range(start_epoch, job["epochs"] + 1):
        next_epoch = epoch
        generator = torch.Generator().manual_seed(job["seed"] + epoch)
        loader = DataLoader(EpochDataset(dataset, job["seed"], epoch), batch_size=job["batch_size"],
                            shuffle=True, num_workers=job["num_workers"], pin_memory=device.type == "cuda",
                            generator=generator)
        model.train()
        # Schedule depends on epoch, not account wall-clock or total optimizer updates.
        warmup = min(1.0, epoch / 2.0)
        progress = min(1.0, max(0.0, (epoch - 2) / 18.0))
        lr = warmup * (1e-6 + (job["lr"] - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_total, epoch_batches = 0.0, 0
        for index, batch in enumerate(loader):
            if epoch == start_epoch and index < skip_batches:
                continue
            # Independent severity RNG keeps every variant's data/condition schedule matched.
            sev_rng = torch.Generator().manual_seed(job["seed"] + epoch * 1000003 + index)
            severity = torch.rand(batch["video"].shape[0], generator=sev_rng).to(device)
            window_start = (index // job["grad_accum_steps"]) * job["grad_accum_steps"]
            window_size = min(job["grad_accum_steps"], len(loader) - window_start)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                losses = batch_loss(model, codec, criterion, teacher, batch, job, device, severity)
                loss = losses["loss_total"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; last completed checkpoint remains available")
            scaler.scale(loss / window_size).backward()
            epoch_total += float(loss.detach())
            epoch_batches += 1
            next_batch = index + 1
            boundary = next_batch % job["grad_accum_steps"] == 0 or next_batch == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite model gradient; refusing an invalid optimizer step")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                elapsed = time.monotonic() - started
                print(f"V3_STEP job={job['job_id']} epoch={epoch} batch={next_batch}/{len(loader)} update={updates} loss={loss.item():.5f} elapsed={elapsed:.1f}s", flush=True)
                if canary and updates >= 2:
                    status = "canary_passed"
                    phase = "canary"
                    snapshot()
                    restored = restore_checkpoint(checkpoint_path, job, model, optimizer, scaler)
                    if restored["optimizer_updates"] != updates:
                        raise RuntimeError("Canary checkpoint update count did not round-trip")
                    report = {"job_id": job["job_id"], "status": status, "optimizer_updates": updates,
                              "seconds": elapsed, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                              "checkpoint_sha256": sha256_file(checkpoint_path),
                              "checkpoint_restored": True, "provenance": provenance}
                    (output / f"{job['job_id']}.canary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    print("V3_CANARY_OK " + json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)
                    return report
                if elapsed >= job["max_hours"] * 3600:
                    status = "time_budget_exhausted"
                    snapshot()
                    print("V3_INCOMPLETE checkpoint=" + str(checkpoint_path), flush=True)
                    return {"status": status, "checkpoint": str(checkpoint_path)}
                if updates <= 2 or updates % 20 == 0:
                    snapshot()
        phase = "validation"
        snapshot()
        try:
            value = validation_loss(model, codec, criterion, teacher, val_dataset, job, device,
                                    deadline=started + job["max_hours"] * 3600)
        except TimeoutError:
            status = "time_budget_exhausted"
            snapshot()
            print("V3_INCOMPLETE validation_checkpoint=" + str(checkpoint_path), flush=True)
            return {"status": status, "checkpoint": str(checkpoint_path)}
        row = {"epoch": epoch, "train_loss": epoch_total / max(1, epoch_batches),
               "train_batches_this_session": epoch_batches, "val_loss": value, "lr": lr}
        history.append(row)
        if value < best_loss - job["min_delta"]:
            best_loss, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        next_epoch, next_batch = epoch + 1, 0
        phase = "train"
        if epoch >= job["min_epochs"] and epoch - best_epoch >= job["patience"]:
            status = "converged"
        elif epoch == job["epochs"]:
            status = "completed"
        snapshot()
        print("V3_EPOCH " + json.dumps(row), flush=True)
        if status != "running":
            break
    return {"status": status, "checkpoint": str(checkpoint_path)}


@torch.no_grad()
def evaluate(job, checkpoint_path, data_root, output_dir, use_yolo=False):
    validate_job(job)
    if current_revision() != job["source_revision"]:
        raise RuntimeError("Evaluation source revision differs from training job")
    dataset = load_real_dataset(job, data_root, "test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_resume(checkpoint, job)
    if checkpoint.get("status") not in ("completed", "converged") or not checkpoint.get("best_state"):
        raise ValueError("Final evaluation requires a completed job with validation-selected weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(job["variant"]).to(device).eval()
    model.load_state_dict(checkpoint["best_state"], strict=True)
    teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=use_yolo).to(device).eval()
    require_teacher(teacher, require_yolo=use_yolo)
    codec = HybridX265STE(use_real_x265_in_forward=True).to(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"{job['job_id']}.records.jsonl"
    metadata_path = output / f"{job['job_id']}.eval.json"
    if metadata_path.exists():
        raise FileExistsError("Evaluation already exists")
    with records_path.open("x", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            sample = dataset[index]
            video = sample["video"].unsqueeze(0).to(device)
            for crf in CRFS:
                severity = torch.tensor([(crf - 22) / 25.0], device=device)
                edited = model(video, torch.zeros(1, dtype=torch.long, device=device), severity,
                               disable_temporal=job["settings"]["disable_temporal"])
                for method, images, offsets in (("anchor_x265", video, None),
                                                ("model", edited["i_edit"], edited["ctu_qp_offsets"])):
                    decoded = codec.evaluate_real_x265(images, crf, offsets)
                    metrics = teacher.evaluate_task_accuracy(video, decoded["video_recon"], sample["m_det"].unsqueeze(0).to(device),
                        gt_boxes=sample["gt_boxes"].unsqueeze(0).to(device),
                        ignore_boxes=sample["ignore_boxes"].unsqueeze(0).to(device),
                        require_real_detector=True, return_raw_detections=True)
                    raw = {k: [v.tolist() for v in values] for k, values in metrics["raw_detections"].items()
                           if k in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores", "yolo_pred_boxes", "yolo_pred_scores")}
                    record = {"method": method, "crf": crf, "frame_ids": sample["frame_ids"],
                              "packet_sizes": decoded["packet_sizes"], "frame_types": decoded["frame_types"],
                              "height": video.shape[-2], "width": video.shape[-1], "raw_detections": raw,
                              "task_roi_psnr_db": metrics["task_roi_psnr_db"]}
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            print(f"V3_EVAL {job['job_id']} clip={index+1}/{len(dataset)}", flush=True)
    metadata = {"schema_version": 3, "job": job, "checkpoint_status": checkpoint["status"],
                "checkpoint_sha256": sha256_file(checkpoint_path), "records_file": records_path.name,
                "records_sha256": sha256_file(records_path), "test_data": data_fingerprint(dataset),
                "expected_frame_ids": dataset.physical_frame_manifest(), "metric_policy": METRIC_POLICY,
                "evaluation_yolo": use_yolo, "provenance": execution_provenance(), "final_ready": False}
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "canary"), required=True)
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-yolo", action="store_true")
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    if args.mode == "eval":
        if not args.checkpoint:
            parser.error("--checkpoint is required for eval")
        result = evaluate(job, args.checkpoint, args.data_root, args.output_dir, args.eval_yolo)
    else:
        result = train(job, args.data_root, args.output_dir, args.resume, canary=args.mode == "canary")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
