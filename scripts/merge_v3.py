"""Validate and aggregate independently seeded, paired schema-3 evaluations.

AP is recomputed over all decoded frames using the preserved dataset-level
metric. Rates use summed video packet bytes and actual pixel denominators.
PCHIP is primary; cubic is reported as sensitivity. Seed dispersion is the
sample standard deviation, unavailable when fewer than two seeds are usable.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_protocol import (
    CRFS, EXPECTED_TEST_FRAME_IDS, METRIC_POLICY, restore_raw_detections, sha256_file,
)
from scripts.merge_shards import summarize_records
from scripts.rd_metrics import compare_bd_rates
from scripts.run_v3 import validate_job
from scripts.v3_protocol import SEEDS, VARIANTS, build_job_matrix


METHODS = ("anchor_x265", "model")
PAIRED_EDGES = {
    "temporal_without_qg": ("spatial", "temporal"),
    "temporal_with_qg": ("spatial_qp", "temporal_qp"),
    "qg_without_temporal": ("spatial", "spatial_qp"),
    "qg_with_temporal": ("temporal", "temporal_qp"),
    "real_codec_vs_proxy": ("temporal_qp_proxy", "temporal_qp"),
}


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _read_json(text):
    return json.loads(text, parse_constant=_reject_constant)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing or invalid {name} SHA-256")


def _fingerprint(value):
    if not isinstance(value, dict) or type(value.get("physical_frames")) is not int or value["physical_frames"] != len(EXPECTED_TEST_FRAME_IDS):
        raise ValueError("Test data fingerprint has incorrect physical frame coverage")
    _digest(value.get("images_sha256"), "test images")
    annotations = value.get("gt_sha256")
    sequences = {frame.split("/")[0] for frame in EXPECTED_TEST_FRAME_IDS}
    if not isinstance(annotations, dict) or set(annotations) != sequences:
        raise ValueError("Test GT fingerprint sequence coverage differs")
    for digest in annotations.values():
        _digest(digest, "test GT")


def _source_fingerprints(provenance):
    sources = provenance.get("source_sha256") if isinstance(provenance, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing source fingerprints")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Source fingerprint paths must be repository-relative")
        _digest(digest, "source")
    return sources


def _metadata(path):
    metadata = _read_json(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 3:
        raise ValueError("Expected a schema-3 evaluation metadata file")
    job = metadata.get("job")
    if not isinstance(job, dict):
        raise ValueError("Missing schema-3 job")
    try:
        validate_job(job)
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed schema-3 job configuration") from error
    if metadata.get("checkpoint_status") not in ("completed", "converged"):
        raise ValueError("Final evaluation requires a completed or converged checkpoint")
    _digest(metadata.get("checkpoint_sha256"), "checkpoint")
    _digest(metadata.get("records_sha256"), "records")
    name = metadata.get("records_file")
    if not isinstance(name, str) or not name.endswith(".records.jsonl") or any(c in name for c in ("/", "\\", ":")) or Path(name).name != name:
        raise ValueError("records_file must be a safe basename ending in .records.jsonl")
    records_path = path.parent / name
    if records_path.is_symlink() or not records_path.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError("Unsafe records path")
    if not records_path.is_file() or sha256_file(records_path) != metadata["records_sha256"]:
        raise ValueError("Evaluation records hash mismatch or missing records file")
    if metadata.get("expected_frame_ids") != EXPECTED_TEST_FRAME_IDS:
        raise ValueError("Expected physical frame manifest differs from the held-out protocol")
    if metadata.get("metric_policy") != METRIC_POLICY:
        raise ValueError("Metric policy differs from dataset-level pedestrian AP")
    if type(metadata.get("evaluation_yolo")) is not bool:
        raise ValueError("evaluation_yolo must explicitly be a boolean")
    _fingerprint(metadata.get("test_data"))
    sources = _source_fingerprints(metadata.get("provenance"))
    # run_id identifies an execution, not an experimental intervention.
    configuration = {key: value for key, value in job.items()
                     if key not in ("job_id", "variant", "seed", "settings", "run_id")}
    cohort = {"configuration": configuration, "source_sha256": sources,
              "test_data": metadata["test_data"], "evaluation_yolo": metadata["evaluation_yolo"]}
    return metadata, records_path, cohort


def _validated_records(metadata, records_path, ground_truth):
    job = metadata["job"]
    clip_len = job["clip_len"]
    if len(EXPECTED_TEST_FRAME_IDS) % clip_len:
        raise ValueError("Expected physical frame manifest contains an incomplete clip")
    partitions = {tuple(EXPECTED_TEST_FRAME_IDS[i:i + clip_len])
                  for i in range(0, len(EXPECTED_TEST_FRAME_IDS), clip_len)}
    groups = {(method, crf): {"frames": set(), "records": []} for method in METHODS for crf in CRFS}
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = _read_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid evaluation record at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError("Evaluation records must be objects")
            method, crf = record.get("method"), record.get("crf")
            if method not in METHODS or type(crf) is not int or crf not in CRFS:
                raise ValueError("Unknown evaluation method or CRF")
            frames = record.get("frame_ids")
            if not isinstance(frames, list) or any(not isinstance(frame, str) for frame in frames) or tuple(frames) not in partitions:
                raise ValueError("Physical frame coverage or paired clip partition differs")
            group = groups[method, crf]
            if group["frames"].intersection(frames):
                raise ValueError("Duplicate physical frame in an evaluation curve")
            sizes = record.get("packet_sizes")
            if not isinstance(sizes, list) or len(sizes) != clip_len or any(type(size) is not int or size <= 0 for size in sizes):
                raise ValueError("Packet sizes must be positive integer byte counts for every frame")
            if record.get("frame_types") != ["I"] + ["P"] * (clip_len - 1):
                raise ValueError("Decoded frame structure differs from the I/P codec protocol")
            height, width = record.get("height"), record.get("width")
            if type(height) is not int or type(width) is not int or height != job["img_size"] or width != job["img_size"]:
                raise ValueError("Record pixel dimensions differ from the matched job")
            roi = record.get("task_roi_psnr_db")
            if roi is not None and (type(roi) not in (int, float) or not math.isfinite(roi)):
                raise ValueError("ROI PSNR must be finite when available")
            raw = record.get("raw_detections")
            if not isinstance(raw, dict):
                raise ValueError("Missing raw detections for global AP")
            yolo_keys = {"yolo_pred_boxes", "yolo_pred_scores"}
            present_yolo = yolo_keys.intersection(raw)
            if metadata["evaluation_yolo"] and present_yolo != yolo_keys:
                raise ValueError("Declared evaluation YOLO detections are missing")
            if not metadata["evaluation_yolo"] and present_yolo:
                raise ValueError("Unexpected undeclared YOLO detections")
            restore_raw_detections(raw, clip_len)
            for index, frame in enumerate(frames):
                gt = (raw["gt_boxes"][index], raw["ignore_boxes"][index], height, width)
                if ground_truth.setdefault(frame, gt) != gt:
                    raise ValueError("Paired GT/ignore boxes differ across methods, CRFs or jobs")
            group["frames"].update(frames)
            group["records"].append(record)
    expected = set(EXPECTED_TEST_FRAME_IDS)
    for group in groups.values():
        if group["frames"] != expected:
            raise ValueError("Incomplete paired physical frame coverage for an evaluation curve")
        group["records"].sort(key=lambda record: tuple(record["frame_ids"]))
    return groups


def _comparisons(test, anchor, use_yolo):
    accuracy_keys = ["gt_ap50_pct", "gt_ap50_95_pct"]
    if use_yolo:
        accuracy_keys += ["yolo_gt_ap50_pct", "yolo_gt_ap50_95_pct"]
    result = {}
    for rate_key, prefix in (("total_bpp", "tot"), ("p_frame_bpp", "p")):
        for accuracy in accuracy_keys:
            result[f"bdbr_{prefix}_{accuracy}"] = compare_bd_rates(
                [anchor[str(crf)][rate_key] for crf in CRFS],
                [anchor[str(crf)][accuracy] for crf in CRFS],
                [test[str(crf)][rate_key] for crf in CRFS],
                [test[str(crf)][accuracy] for crf in CRFS],
            )
    return result


def _statistics(values):
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) >= 2 else None,
            "n": len(values)}


def _aggregate(jobs, matrix, use_yolo):
    # The summary reports independent training replicates, never a weight soup.
    field_names = [f"bdbr_{prefix}_{metric}" for prefix in ("tot", "p")
                   for metric in (["gt_ap50_pct", "gt_ap50_95_pct"] +
                                  (["yolo_gt_ap50_pct", "yolo_gt_ap50_95_pct"] if use_yolo else []))]
    result = {}
    for variant in VARIANTS:
        variant_jobs = [job for job in matrix if job["variant"] == variant]
        fields = {}
        for name in field_names:
            primary, cubic, differences, unavailable = [], [], [], []
            for job in variant_jobs:
                current = jobs.get(job["job_id"])
                comparison = current["methods"]["model"]["bd_rate"][name] if current else None
                if comparison and comparison["primary"] is not None:
                    primary.append(comparison["primary"]["value_pct"])
                else:
                    unavailable.append(job["seed"])
                if comparison and comparison["cubic_sensitivity"] is not None:
                    cubic.append(comparison["cubic_sensitivity"]["value_pct"])
                if comparison and comparison["difference_pp"] is not None:
                    differences.append(comparison["difference_pp"])
            fields[name] = {**_statistics(primary), "expected_n": len(variant_jobs),
                            "unavailable_seeds": unavailable,
                            "cubic_sensitivity": _statistics(cubic),
                            "difference_pp": _statistics(differences)}
        result[variant] = fields
    return result


def merge_evaluations(metadata_paths, output_path=None, seeds=SEEDS, allow_partial=False):
    """Merge an exact variant/seed matrix, or explicitly label partial progress.

    All present files must pass full provenance and paired-coverage validation,
    including when ``allow_partial`` is true. Invalid files never create output.
    Existing output files are preserved. Paths in the result are basenames or
    repository-relative source names, so results are portable between accounts.
    """
    matrix = build_job_matrix(seeds)
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be a boolean")
    expected = {job["job_id"] for job in matrix}
    if isinstance(metadata_paths, (str, Path)):
        metadata_paths = [metadata_paths]
    entries, jobs, ground_truth = {}, {}, {}
    common = None
    for value in metadata_paths:
        path = Path(value)
        metadata, records_path, cohort = _metadata(path)
        identity = metadata["job"]["job_id"]
        if identity not in expected:
            raise ValueError("Evaluation job is outside the requested variant/seed matrix")
        if identity in entries:
            raise ValueError("Duplicate evaluation job identity")
        if common is None:
            common = cohort
        elif common != cohort:
            raise ValueError("Evaluation jobs have mismatched configuration, source or data fingerprints")
        entries[identity] = (path, metadata, records_path)
    missing = [job["job_id"] for job in matrix if job["job_id"] not in entries]
    if missing and not allow_partial:
        raise ValueError(f"Incomplete evaluation matrix; missing {len(missing)} requested jobs")
    for identity, (path, metadata, records_path) in sorted(entries.items()):
        groups = _validated_records(metadata, records_path, ground_truth)
        per_method = {method: {str(crf): summarize_records(groups[method, crf]["records"])
                               for crf in CRFS} for method in METHODS}
        job = metadata["job"]
        jobs[identity] = {"variant": job["variant"], "seed": job["seed"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "sources": {"metadata": {"name": path.name, "sha256": sha256_file(path)},
                        "records": {"name": records_path.name, "sha256": metadata["records_sha256"]}},
            "methods": {method: {"per_crf": per_method[method],
                "bd_rate": _comparisons(per_method[method], per_method["anchor_x265"], metadata["evaluation_yolo"])}
                for method in METHODS}}
    use_yolo = common["evaluation_yolo"] if common else False
    paired = {}
    for seed in dict.fromkeys(job["seed"] for job in matrix):
        edges = {}
        for name, (anchor_variant, test_variant) in PAIRED_EDGES.items():
            anchor = jobs.get(f"{anchor_variant}_seed{seed}")
            test = jobs.get(f"{test_variant}_seed{seed}")
            if anchor is not None and test is not None:
                edges[name] = {"anchor_variant": anchor_variant, "test_variant": test_variant,
                    "bd_rate": _comparisons(test["methods"]["model"]["per_crf"],
                                            anchor["methods"]["model"]["per_crf"], use_yolo)}
        paired[str(seed)] = edges
    result = {"schema_version": 3,
        "eligibility": {"final_ready": not missing, "complete_jobs": len(jobs),
                        "expected_jobs": len(matrix), "missing_job_ids": missing,
                        "frame_count": len(ground_truth)},
        "source_revision": common["configuration"]["source_revision"] if common else None,
        "source_sha256": common["source_sha256"] if common else {},
        "test_data": common["test_data"] if common else None,
        "metric_policy": METRIC_POLICY, "evaluation_yolo": use_yolo,
        "aggregation_policy": "independent_seed_mean_and_sample_std; std_unavailable_for_n_less_than_2",
        "jobs": jobs, "aggregate_variants": _aggregate(jobs, matrix, use_yolo),
        "paired_comparisons": paired}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args(argv)
    paths = sorted(Path(args.input_dir).rglob("*.eval.json"))
    result = merge_evaluations(paths, args.output, seeds=args.seeds, allow_partial=args.allow_partial)
    print(json.dumps(result["eligibility"], allow_nan=False))


if __name__ == "__main__":
    main()
