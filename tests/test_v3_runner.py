"""Protect real-data, matched-budget and restart contracts of the v3 runner."""
import copy
import tempfile
import time
import unittest
from pathlib import Path

import torch

from scripts import run_v3


class V3RunnerTests(unittest.TestCase):
    def test_amp_overflow_skips_update_backs_off_scale_and_recovers(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
        loss = (model(torch.ones(1, 1)) * 1e35).sum()
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        result = run_v3.optimizer_step(model, optimizer, scaler)
        self.assertFalse(result["updated"])
        self.assertEqual(model.weight.item(), 1.0)
        self.assertEqual(scaler.get_scale(), 32768.0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        scaler.scale(model(torch.ones(1, 1)).square().sum()).backward()
        result = run_v3.optimizer_step(model, optimizer, scaler)
        self.assertTrue(result["updated"])
        self.assertLess(model.weight.item(), 1.0)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_unscaled_nonfinite_gradient_is_still_fatal(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        model(torch.ones(1, 1)).sum().backward()
        original = model.weight.detach().clone()
        model.weight.grad.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            run_v3.optimizer_step(model, optimizer, scaler)
        self.assertTrue(torch.equal(original, model.weight))

    def test_large_finite_gradient_is_clipped_without_norm_overflow(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        (model(torch.ones(1, 1)) * 1e30).sum().backward()
        result = run_v3.optimizer_step(model, optimizer, scaler)
        self.assertTrue(result["updated"])
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertAlmostEqual(model.weight.item(), 0.9, places=5)

    def test_spatial_qg_has_no_temporal_regularization(self):
        criterion = run_v3.make_criterion(run_v3.make_job("spatial_qp", 2026))
        video = torch.rand(1, 4, 3, 32, 32)
        zeros = torch.zeros_like(video)
        masks = torch.zeros(1, 4, 1, 32, 32)
        offsets = torch.tensor([4., 3., 2., 1.]).reshape(1, 4, 1, 1).requires_grad_()
        result = criterion(i_edit=video, i_recon=video, attn_map=masks + 0.2,
                           delta_obj=zeros, delta_bg=zeros, rate_proxy_bpp=torch.ones(1),
                           m_attn=masks, m_det=masks, teacher_fpn_feats={},
                           student_fpn_feats={}, ctu_qp_offsets=offsets)
        self.assertEqual(result["loss_temp"].item(), 0.0)

    def test_validation_expired_deadline_preserves_training_mode(self):
        from scripts.v3_protocol import build_model
        model = build_model("spatial", base_ch=8).train()
        sample = {"video": torch.zeros(4, 3, 32, 32)}
        with self.assertRaises(TimeoutError):
            run_v3.validation_loss(model, None, None, None, [sample],
                                   run_v3.make_job("spatial", 2026),
                                   torch.device("cpu"), deadline=time.monotonic()-1)
        self.assertTrue(model.training)

    def test_canary_checkpoint_readback_restores_weights_and_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        job = run_v3.make_job("temporal_qp", 2026)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"canary.pt"
            run_v3.save_checkpoint(path, {"schema_version": 3, "job": job,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(), "epoch": 1, "next_batch": 4, "history": []})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            result = run_v3.restore_checkpoint(path, job, model, optimizer, scaler)
        self.assertEqual(result["next_batch"], 4)
        self.assertTrue(all(torch.equal(saved[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(optimizer.state)

    def test_epoch_crop_stream_is_independent_of_model_rng_and_worker_order(self):
        class RandomSample(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return torch.rand(3)

        view = run_v3.EpochDataset(RandomSample(), 2026, 1)
        first = view[2]
        torch.rand(400)
        view[0]
        self.assertTrue(torch.equal(first, view[2]))
        self.assertFalse(torch.equal(first, run_v3.EpochDataset(RandomSample(), 2026, 2)[2]))

    def test_main_factorial_jobs_have_matched_training_conditions(self):
        jobs = [run_v3.make_job(v, 2026) for v in
                ("spatial", "spatial_qp", "temporal", "temporal_qp")]
        for field in ("seed", "epochs", "lr", "batch_size", "grad_accum_steps",
                      "img_size", "clip_len", "train_sequences", "val_sequences",
                      "teacher", "codec"):
            self.assertTrue(all(job[field] == jobs[0][field] for job in jobs), field)
        self.assertEqual(jobs[0]["teacher"], "fasterrcnn_resnet50_fpn_v2")
        self.assertEqual(jobs[0]["schema_version"], 3)

    def test_invalid_budget_or_tampered_settings_cannot_start_a_run(self):
        for override in ({"epochs": 0}, {"batch_size": True}, {"max_hours": -1},
                         {"img_size": 33}, {"clip_len": 1}, {"lr": float("nan")},
                         {"min_epochs": 40}, {"unknown_option": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                run_v3.make_job("temporal_qp", 2026, **override)
        job = run_v3.make_job("spatial", 2026)
        job["settings"]["use_real_x265_ste"] = False
        with self.assertRaisesRegex(ValueError, "settings"):
            run_v3.validate_job(job)

    def test_empty_real_dataset_fails_without_synthetic_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "val", "test"):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "real MOT17"):
                    run_v3.load_real_dataset(run_v3.make_job("spatial", 2026), root, split)

    def test_teacher_must_be_real_and_training_must_not_use_yolo(self):
        from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
        fallback = FrozenFPNDetectorTeacher(pretrained=False)
        with self.assertRaisesRegex(RuntimeError, "pretrained"):
            run_v3.require_teacher(fallback, training=True)

    def test_v2_or_other_job_checkpoint_cannot_resume_v3(self):
        job = run_v3.make_job("temporal_qp", 2026)
        for checkpoint in ({"schema_version": 2},
                           {"schema_version": 3, "job": run_v3.make_job("spatial", 2026)}):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                run_v3.validate_resume(checkpoint, job)

    def test_checkpoint_atomic_write_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            run_v3.save_checkpoint(path, {"schema_version": 3, "weight": torch.tensor([2.0])})
            run_v3.save_checkpoint(path, {"schema_version": 3, "weight": torch.tensor([3.0])})
            actual = torch.load(path, weights_only=True)
            self.assertEqual(actual["weight"].item(), 3.0)
            self.assertEqual([p.name for p in path.parent.iterdir()], ["last.pt"])

    def test_proxy_training_loss_reaches_model_parameters(self):
        from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
        from scripts.v3_protocol import build_model
        from video_codecs.hybrid_x265_ste import HybridX265STE
        from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
        torch.manual_seed(41)
        job = run_v3.make_job("temporal_qp_proxy", 2026)
        model = build_model(job["variant"], base_ch=8)
        video = torch.rand(1, 2, 3, 32, 32)
        mask = torch.zeros(1, 2, 1, 32, 32)
        mask[..., 8:24, 8:24] = 1
        batch = {"video": video, "m_attn": mask, "m_det": mask}
        losses = run_v3.batch_loss(model, HybridX265STE(),
                                   VCMTemporalPreEditLoss(),
                                   FrozenFPNDetectorTeacher(pretrained=False),
                                   batch, job, torch.device("cpu"),
                                   severity=torch.tensor([0.4]))
        losses["loss_total"].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)


if __name__ == "__main__":
    unittest.main()
