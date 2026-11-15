"""Exercise v3 aggregation using real dataset AP and packet accounting."""

import copy
import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiment_protocol import CRFS, METRIC_POLICY, sha256_file
from scripts.run_v3 import make_job
from scripts.v3_protocol import VARIANTS


class MergeV3Tests(unittest.TestCase):
    def setUp(self):
        try:
            self.merger = importlib.import_module("scripts.merge_v3")
        except ModuleNotFoundError as error:
            if error.name != "scripts.merge_v3":
                raise
            self.fail("The independent schema-3 merger has not been implemented")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.ids = [f"MOT17-11/{i:06d}" for i in range(1, 9)]
        patched = patch.object(self.merger, "EXPECTED_TEST_FRAME_IDS", self.ids)
        patched.start()
        self.addCleanup(patched.stop)

    def collection(self, seeds=(2026,), variants=VARIANTS, scales=None):
        paths = []
        for variant in variants:
            for seed in seeds:
                job = make_job(variant, seed, img_size=32, num_workers=0, source_revision="a" * 40)
                scale = (scales or {}).get((variant, seed), 0.8)
                records = []
                for crf_index, crf in enumerate(CRFS):
                    for method in ("anchor_x265", "model"):
                        packet = round([1000, 500, 250, 125][crf_index] * (scale if method == "model" else 1))
                        for clip in range(2):
                            correct = 4 - crf_index
                            raw = {"gt_boxes": [[[0, 0, 10, 10]]] * 4,
                                   "ignore_boxes": [[] for _ in range(4)],
                                   "pred_boxes": [[ [0, 0, 10, 10] ] if i < correct else [] for i in range(4)],
                                   "pred_scores": [[0.9] if i < correct else [] for i in range(4)]}
                            records.append({"method": method, "crf": crf,
                                "frame_ids": self.ids[clip * 4:(clip + 1) * 4],
                                "packet_sizes": [packet] * 4, "frame_types": ["I", "P", "P", "P"],
                                "height": 32, "width": 32, "raw_detections": raw, "task_roi_psnr_db": 25.0})
                records_path = self.directory / f"{job['job_id']}.records.jsonl"
                records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
                metadata = {"schema_version": 3, "job": job, "checkpoint_status": "completed",
                    "checkpoint_sha256": "c" * 64, "records_file": records_path.name,
                    "records_sha256": sha256_file(records_path),
                    "test_data": {"physical_frames": 8, "images_sha256": "d" * 64,
                                  "gt_sha256": {"MOT17-11": "e" * 64}},
                    "expected_frame_ids": self.ids, "metric_policy": copy.deepcopy(METRIC_POLICY),
                    "evaluation_yolo": False, "provenance": {"source_sha256": {"scripts/run_v3.py": "f" * 64},
                    "python": "3.11.0", "torch": "2.5.1"}, "final_ready": False}
                path = self.directory / f"{job['job_id']}.eval.json"
                path.write_text(json.dumps(metadata), encoding="utf-8")
                paths.append(path)
        return paths

    def change(self, path, metadata_change=None, records_change=None, rehash=True):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        records_path = path.parent / metadata["records_file"]
        if records_change:
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            records_change(records)
            records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            if rehash:
                metadata["records_sha256"] = sha256_file(records_path)
        if metadata_change:
            metadata_change(metadata)
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_complete_requested_matrix_computes_real_rates_and_bd_rate(self):
        paths = self.collection()
        output = self.directory / "merged.json"
        result = self.merger.merge_evaluations(paths, output, seeds=(2026,))
        self.assertEqual(result["schema_version"], 3)
        self.assertTrue(result["eligibility"]["final_ready"])
        self.assertEqual(result["eligibility"]["complete_jobs"], 6)
        self.assertEqual(result["eligibility"]["expected_jobs"], 6)
        self.assertEqual(result["eligibility"]["frame_count"], 8)
        methods = result["jobs"]["spatial_seed2026"]["methods"]
        self.assertAlmostEqual(methods["anchor_x265"]["per_crf"]["22"]["total_bpp"], 8000 / 1024)
        self.assertAlmostEqual(methods["model"]["bd_rate"]["bdbr_tot_gt_ap50_pct"]["primary"]["value_pct"], -20)
        self.assertNotIn("yolo_gt_ap50_pct", methods["model"]["per_crf"]["22"])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
        self.assertNotIn(str(self.directory), json.dumps(result))

    def test_missing_seed_replicate_cannot_be_final(self):
        paths = self.collection()
        output = self.directory / "incomplete.json"
        with self.assertRaisesRegex(ValueError, "(?i)(missing|incomplete)"):
            self.merger.merge_evaluations(paths, output, seeds=(2026, 2027))
        self.assertFalse(output.exists())
        partial = self.merger.merge_evaluations(paths, seeds=(2026, 2027), allow_partial=True)
        self.assertFalse(partial["eligibility"]["final_ready"])
        self.assertEqual(len(partial["eligibility"]["missing_job_ids"]), 6)
        self.assertIn("temporal_qp_seed2027", partial["eligibility"]["missing_job_ids"])

    def test_duplicate_jobs_and_invalid_seed_requests_are_rejected(self):
        paths = self.collection(variants=("spatial",))
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            self.merger.merge_evaluations(paths + paths, seeds=(2026,), allow_partial=True)
        for seeds in ((True,), (2026, 2026), (), ("2026",)):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    self.merger.merge_evaluations(paths, seeds=seeds, allow_partial=True)

    def test_duplicate_or_missing_physical_frames_rejected_even_in_partial_mode(self):
        paths = self.collection(variants=("spatial",))
        original = Path(paths[0]).read_text(encoding="utf-8")
        original_records = (self.directory / "spatial_seed2026.records.jsonl").read_text(encoding="utf-8")
        for mutation in (lambda rows: rows.append(copy.deepcopy(rows[0])), lambda rows: rows.pop()):
            paths[0].write_text(original, encoding="utf-8")
            (self.directory / "spatial_seed2026.records.jsonl").write_text(original_records, encoding="utf-8")
            self.change(paths[0], records_change=mutation)
            with self.assertRaisesRegex(ValueError, "(?i)(duplicate|coverage|missing|incomplete)"):
                self.merger.merge_evaluations(paths, seeds=(2026,), allow_partial=True)

    def test_mismatched_gt_is_rejected_across_methods(self):
        paths = self.collection(variants=("spatial",))
        self.change(paths[0], records_change=lambda rows: rows[2]["raw_detections"]["gt_boxes"].__setitem__(0, [[0, 0, 9, 9]]))
        with self.assertRaisesRegex(ValueError, "(?i)(GT|ground.truth)"):
            self.merger.merge_evaluations(paths, seeds=(2026,), allow_partial=True)

    def test_changed_configuration_source_or_test_data_is_rejected(self):
        changes = (
            lambda meta: meta["job"].update(lr=0.001),
            lambda meta: meta["job"].update(source_revision="b" * 40),
            lambda meta: meta["provenance"]["source_sha256"].update({"scripts/run_v3.py": "0" * 64}),
            lambda meta: meta["test_data"].update(images_sha256="0" * 64),
        )
        paths = self.collection(variants=("spatial", "temporal"))
        original = paths[1].read_text(encoding="utf-8")
        for mutation in changes:
            with self.subTest(mutation=mutation):
                paths[1].write_text(original, encoding="utf-8")
                self.change(paths[1], metadata_change=mutation)
                with self.assertRaisesRegex(ValueError, "(?i)(match|differ|configuration|fingerprint)"):
                    self.merger.merge_evaluations(paths, seeds=(2026,), allow_partial=True)

    def test_record_hash_and_unsafe_record_paths_fail_before_output(self):
        paths = self.collection(variants=("spatial",))
        original = paths[0].read_text(encoding="utf-8")
        for mutation in (
            lambda meta: meta.update(records_sha256="0" * 64),
            lambda meta: meta.update(records_file="../outside.records.jsonl"),
            lambda meta: meta.update(records_file="C:\\outside.records.jsonl"),
        ):
            with self.subTest(mutation=mutation):
                paths[0].write_text(original, encoding="utf-8")
                self.change(paths[0], metadata_change=mutation)
                output = self.directory / "rejected.json"
                with self.assertRaisesRegex(ValueError, "(?i)(hash|path|basename)"):
                    self.merger.merge_evaluations(paths, output, seeds=(2026,), allow_partial=True)
                self.assertFalse(output.exists())

    def test_incomplete_checkpoint_and_missing_declared_yolo_are_rejected(self):
        paths = self.collection(variants=("spatial",))
        original = paths[0].read_text(encoding="utf-8")
        for mutation in (
            lambda meta: meta.update(checkpoint_status="time_budget_exhausted"),
            lambda meta: meta.update(evaluation_yolo=True),
            lambda meta: meta["job"].update(seed=True),
        ):
            paths[0].write_text(original, encoding="utf-8")
            self.change(paths[0], metadata_change=mutation)
            with self.assertRaises(ValueError):
                self.merger.merge_evaluations(paths, seeds=(2026,), allow_partial=True)

    def test_dataset_ap_uses_global_detection_ranking_and_raw_packet_sums(self):
        paths = self.collection(variants=("spatial",))

        def globally_ranked_records(rows):
            for row in rows:
                first = row["frame_ids"][0] == self.ids[0]
                sizes = [10, 20, 30, 40] if first else [50, 60, 70, 80]
                row["packet_sizes"] = [size * (1 + CRFS.index(row["crf"])) for size in sizes]
                row["raw_detections"]["pred_boxes"] = [[[0, 0, 10, 10] if first else [20, 20, 30, 30]]] * 4
                row["raw_detections"]["pred_scores"] = [[0.5 if first else 0.9]] * 4

        self.change(paths[0], records_change=globally_ranked_records)
        result = self.merger.merge_evaluations(paths, seeds=(2026,), allow_partial=True)
        method = result["jobs"]["spatial_seed2026"]["methods"]["model"]
        summary = method["per_crf"]["22"]
        # Four high-score FPs precede four TPs: precision=.5, recall reaches .5.
        # The shared dataset AP implementation reports three decimal places.
        self.assertAlmostEqual(summary["gt_ap50_pct"], 100 * 25.5 / 101, places=3)
        self.assertNotAlmostEqual(summary["gt_ap50_pct"], 50.0)
        self.assertEqual(summary["packet_bytes"], 360)
        self.assertAlmostEqual(summary["total_bpp"], 360 * 8 / (8 * 1024))
        self.assertAlmostEqual(summary["p_frame_bpp"], 300 * 8 / (6 * 1024))
        unavailable = method["bd_rate"]["bdbr_tot_gt_ap50_pct"]
        self.assertIsNone(unavailable["primary"])
        self.assertRegex(unavailable["unavailable_reason"], "(?i)(plateau|duplicate accuracy)")

    def test_independent_seed_statistics_and_direct_paired_factor_comparison(self):
        scales = {(variant, seed): (0.8 if seed == 2026 else 0.6)
                  for variant in VARIANTS for seed in (2026, 2027)}
        scales[("temporal", 2026)] = 0.6
        paths = self.collection(seeds=(2026, 2027), scales=scales)
        result = self.merger.merge_evaluations(paths, seeds=(2026, 2027))
        statistic = result["aggregate_variants"]["spatial"]["bdbr_tot_gt_ap50_pct"]
        self.assertAlmostEqual(statistic["mean"], -30)
        self.assertAlmostEqual(statistic["std"], math.sqrt(200))
        self.assertEqual(statistic["n"], 2)
        paired = result["paired_comparisons"]["2026"]["temporal_without_qg"]
        self.assertEqual(paired["anchor_variant"], "spatial")
        self.assertEqual(paired["test_variant"], "temporal")
        # .6/.8 - 1 = -25%; subtracting the two anchor savings gives -20%.
        self.assertAlmostEqual(paired["bd_rate"]["bdbr_tot_gt_ap50_pct"]["primary"]["value_pct"], -25)


if __name__ == "__main__":
    unittest.main()
