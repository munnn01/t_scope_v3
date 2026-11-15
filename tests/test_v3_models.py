"""Contracts for the separate v3 experiment registry and paper adaptation."""
from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F


class V3ProtocolTests(unittest.TestCase):
    def protocol(self):
        self.assertIsNotNone(importlib.util.find_spec("scripts.v3_protocol"), "v3 protocol is not implemented")
        from scripts import v3_protocol

        return v3_protocol

    def test_factorial_settings_and_proxy_control(self):
        protocol = self.protocol()
        self.assertEqual(tuple(protocol.VARIANTS), (
            "spatial", "spatial_qp", "temporal", "temporal_qp",
            "temporal_qp_proxy", "scope_paper_adapted",
        ))
        self.assertEqual(protocol.SEEDS, (2026, 2027, 2028))
        for variant, temporal, qp in (
            ("spatial", False, False), ("spatial_qp", False, True),
            ("temporal", True, False), ("temporal_qp", True, True),
        ):
            with self.subTest(variant=variant):
                self.assertEqual(protocol.variant_settings(variant), {
                    "disable_temporal": not temporal,
                    "max_delta_qp": 4.0 if qp else 0.0,
                    "use_real_x265_ste": True,
                    "lambda_temp_bg": 1.0 if temporal else 0.0,
                    "lambda_temp_obj": 0.5 if temporal else 0.0,
                    "model_kind": "temporal_scope",
                })
        real = protocol.variant_settings("temporal_qp")
        proxy = protocol.variant_settings("temporal_qp_proxy")
        self.assertEqual(proxy, dict(real, use_real_x265_ste=False))
        self.assertEqual(protocol.variant_settings("scope_paper_adapted"), {
            "disable_temporal": True,
            "max_delta_qp": 0.0,
            "use_real_x265_ste": False,
            "lambda_temp_bg": 0.0,
            "lambda_temp_obj": 0.0,
            "model_kind": "scope_paper_adapted",
        })

    def test_main_models_have_matched_initial_weights_and_qg_factor(self):
        protocol = self.protocol()
        from models.temporal_scope import TemporalSCOPE

        expected = None
        for variant in protocol.VARIANTS[:5]:
            torch.manual_seed(2026)
            model = protocol.build_model(variant, base_ch=8, cond_dim=16)
            self.assertIsInstance(model, TemporalSCOPE)
            self.assertEqual(model.max_delta_qp, 4.0 if "qp" in variant else 0.0)
            state = model.state_dict()
            if expected is None:
                expected = {key: value.clone() for key, value in state.items()}
            else:
                self.assertEqual(set(state), set(expected))
                for key in state:
                    torch.testing.assert_close(state[key], expected[key], rtol=0, atol=0)
        from models.scope_reference import ScopePaperAdapted

        self.assertIsInstance(protocol.build_model("scope_paper_adapted"), ScopePaperAdapted)
        with self.assertRaises(ValueError):
            protocol.build_model("spatial", max_delta_qp=4.0)

    def test_job_matrix_contains_complete_replicates_without_shared_settings(self):
        protocol = self.protocol()
        jobs = protocol.build_job_matrix()
        self.assertEqual(len(jobs), 18)
        expected_pairs = [
            (variant, seed) for variant in protocol.VARIANTS for seed in (2026, 2027, 2028)
        ]
        self.assertEqual([(job["variant"], job["seed"]) for job in jobs], expected_pairs)
        self.assertEqual(len({job["job_id"] for job in jobs}), 18)
        for job in jobs:
            self.assertEqual(set(job), {"job_id", "variant", "seed", "settings"})
            self.assertEqual(job["job_id"], f'{job["variant"]}_seed{job["seed"]}')
            self.assertEqual(job["settings"], protocol.variant_settings(job["variant"]))
        jobs[0]["settings"]["max_delta_qp"] = 99
        self.assertEqual(jobs[1]["settings"]["max_delta_qp"], 0.0)
        self.assertEqual(protocol.variant_settings("spatial")["max_delta_qp"], 0.0)
        self.assertEqual(len(protocol.build_job_matrix(seeds=[17])), 6)

    def test_invalid_variant_and_seed_identities_fail_closed(self):
        protocol = self.protocol()
        for bad in ("", "full_tscope", "Temporal", None, True, [], 0):
            with self.subTest(variant=bad), self.assertRaises(ValueError):
                protocol.variant_settings(bad)
            with self.subTest(build=bad), self.assertRaises(ValueError):
                protocol.build_model(bad)
        for bad in ((), [], None, "2026", [True], [1.0], ["2026"], [-1], [2**32], [2026, 2026]):
            with self.subTest(seeds=bad), self.assertRaises(ValueError):
                protocol.build_job_matrix(seeds=bad)


class ScopePaperAdaptedTests(unittest.TestCase):
    def model(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec("models.scope_reference"), "paper adaptation is not implemented")
        from models.scope_reference import ScopePaperAdapted

        return ScopePaperAdapted(**kwargs)

    def test_reference_has_image_only_h4_default_architecture(self):
        model = self.model()
        self.assertEqual(model.enc1.conv1.in_channels, 3)
        self.assertEqual(model.enc1.conv1.out_channels, 48)
        self.assertEqual(model.enc2.conv1.out_channels, 96)
        self.assertEqual(model.bottleneck.conv1.out_channels, 192)
        self.assertEqual(model.conditioner.family_emb.embedding_dim, 8)
        self.assertEqual(model.conditioner.mlp[2].out_features, 64)
        self.assertEqual(model.max_obj_res, 0.09)
        self.assertEqual(model.max_bg_res, 0.03)
        self.assertEqual(model.max_delta_qp, 0.0)
        self.assertFalse(hasattr(model, "temporal_gru"))
        self.assertFalse(hasattr(model, "down3"))

    def test_clip_and_image_shapes_include_zero_qg_and_no_temporal_outputs(self):
        model = self.model(base_ch=8, cond_dim=16)
        family, severity = torch.tensor([0, 0]), torch.tensor([0.2, 0.8])
        for shape, attention, qg in (
            ((2, 3, 3, 35, 39), (2, 3, 1, 35, 39), (2, 3, 2, 2)),
            ((2, 3, 35, 39), (2, 1, 35, 39), (2, 2, 2)),
        ):
            with self.subTest(shape=shape):
                result = model(torch.rand(shape), family, severity)
                self.assertEqual(set(result), {
                    "i_edit", "attn_map", "eff_attn_map", "delta_obj", "delta_bg",
                    "ctu_qp_offsets", "aligned_prev_edit", "aligned_prev_orig", "warp_valid_mask",
                })
                for key in ("i_edit", "delta_obj", "delta_bg"):
                    self.assertEqual(tuple(result[key].shape), shape)
                self.assertEqual(tuple(result["attn_map"].shape), attention)
                self.assertEqual(tuple(result["ctu_qp_offsets"].shape), qg)
                self.assertEqual(int(torch.count_nonzero(result["ctu_qp_offsets"])), 0)
                torch.testing.assert_close(result["attn_map"], result["eff_attn_map"])
                for key in ("aligned_prev_edit", "aligned_prev_orig", "warp_valid_mask"):
                    self.assertIsNone(result[key])
                self.assertGreaterEqual(float(result["i_edit"].min()), 0)
                self.assertLessEqual(float(result["i_edit"].max()), 1)
                self.assertLessEqual(float(result["delta_obj"].abs().max()), 0.09)
                self.assertLessEqual(float(result["delta_bg"].abs().max()), 0.03)

    def test_edit_is_direct_attention_blend_of_bounded_branches(self):
        model = self.model(base_ch=8, cond_dim=16)
        # Constant logits make the expected image independent of the learned backbone.
        with torch.no_grad():
            for head, bias in ((model.attn_head, torch.logit(torch.tensor(0.25))),
                               (model.obj_res_head, torch.atanh(torch.tensor(0.5))),
                               (model.bg_res_head, torch.atanh(torch.tensor(-0.5)))):
                head.weight.zero_()
                head.bias.fill_(bias)
        image = torch.linspace(0, 1, 32 * 32).view(1, 1, 32, 32).expand(1, 3, 32, 32)
        result = model(image, torch.tensor([0]), torch.tensor([0.0]))
        kernel = torch.tensor([1., 4., 6., 4., 1.]) / 16
        weight = torch.outer(kernel, kernel).expand(3, 1, 5, 5)
        blurred = F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), weight, groups=3)
        expected = 0.25 * (image + 0.045).clamp(0, 1) + 0.75 * (blurred - 0.015).clamp(0, 1)
        torch.testing.assert_close(result["i_edit"], expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(result["attn_map"], torch.full_like(result["attn_map"], 0.25))
        torch.testing.assert_close(result["delta_obj"], torch.full_like(image, 0.045))

    def test_framewise_edit_is_invariant_to_order_flags_and_neighbor_frames(self):
        torch.manual_seed(7)
        model = self.model(base_ch=8, cond_dim=16).eval()
        video = torch.rand(2, 3, 3, 32, 32)
        family, severity = torch.tensor([0, 0]), torch.tensor([0.1, 0.9])
        with torch.no_grad():
            clip_result = model(video, family, severity)
            shuffled = model(video[:, [2, 0, 1]], family, severity, reuse_iframe_qp=True, disable_temporal=True)
            for step, original in enumerate((2, 0, 1)):
                image_result = model(video[:, original], family, severity)
                for key in ("i_edit", "attn_map", "delta_obj", "delta_bg"):
                    torch.testing.assert_close(clip_result[key][:, original], image_result[key], rtol=0, atol=1e-6)
                    torch.testing.assert_close(shuffled[key][:, step], image_result[key], rtol=0, atol=1e-6)

    def test_reference_backpropagates_finite_gradients_to_image_and_conditioner(self):
        torch.manual_seed(11)
        model = self.model(base_ch=8, cond_dim=16)
        image = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
        result = model(image, torch.tensor([0]), torch.tensor([0.7]))
        loss = result["i_edit"].square().mean() + result["attn_map"].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
        self.assertGreater(float(model.conditioner.family_emb.weight.grad[0].abs().sum()), 0)

    def test_clip_training_does_not_retain_full_backbone_activations(self):
        model = self.model(base_ch=8, cond_dim=16).train()
        video = torch.rand(1, 3, 3, 32, 32, requires_grad=True)
        retained_elements = []

        def pack(tensor):
            retained_elements.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = model(video, torch.tensor([0]), torch.tensor([0.5]))
        # Clip checkpointing retains inputs/condition, not full-resolution features
        # at every U-Net layer. The allowance includes the conditioning MLP graph.
        self.assertLess(sum(retained_elements), 8 * video.numel())
        result["i_edit"].mean().backward()
        self.assertTrue(torch.isfinite(video.grad).all())


if __name__ == "__main__":
    unittest.main()
