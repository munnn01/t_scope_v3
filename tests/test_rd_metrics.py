"""Independent numerical and validation contracts for the v3 rate/AP metric."""

import importlib
import json
import math
import unittest

import numpy as np


class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.metrics = importlib.import_module("scripts.rd_metrics")
        except ModuleNotFoundError as error:
            if error.name != "scripts.rd_metrics":
                raise
            self.fail("The v3 rate/AP metric API has not been implemented")
        self.rates = [1.0, 2.0, 4.0, 8.0]
        self.accuracy = [10.0, 20.0, 30.0, 40.0]


class RateAccuracyTests(MetricsTestCase):
    def test_constant_twenty_percent_saving_has_negative_sign_and_diagnostics(self):
        result = self.metrics.bd_rate(
            self.rates, self.accuracy, [0.8, 1.6, 3.2, 6.4], self.accuracy
        )
        self.assertAlmostEqual(result["value_pct"], -20.0, places=10)
        self.assertEqual(result["common_accuracy_interval"], [10.0, 40.0])
        self.assertEqual(result["interpolation"], "pchip")
        self.assertEqual(result["operating_points"], {"anchor": 4, "test": 4})
        json.dumps(result, allow_nan=False)

    def test_identical_curve_is_zero_for_both_interpolations(self):
        for interpolation in ("pchip", "cubic"):
            with self.subTest(interpolation=interpolation):
                result = self.metrics.bd_rate(
                    self.rates, self.accuracy, self.rates, self.accuracy,
                    interpolation=interpolation,
                )
                self.assertAlmostEqual(result["value_pct"], 0.0, places=10)

    def test_operating_points_can_be_supplied_in_independent_arbitrary_orders(self):
        result = self.metrics.bd_rate(
            [8, 1, 4, 2], [40, 10, 30, 20],
            [1.6, 6.4, 0.8, 3.2], [20, 40, 10, 30],
        )
        self.assertAlmostEqual(result["value_pct"], -20.0, places=10)

    def test_variable_saving_integrates_only_the_common_accuracy_interval(self):
        # Let x = AP/10. ln(R_anchor) = x; ln(R_test) = 1.2*x - .4.
        # Only x in [2, 4.5] overlaps, so mean log ratio = .2*3.25-.4=.25.
        for interpolation in ("pchip", "cubic"):
            with self.subTest(interpolation=interpolation):
                result = self.metrics.bd_rate(
                    np.exp([1, 2, 3, 4, 5]), [10, 20, 30, 40, 50],
                    np.exp([2, 2.6, 3.8, 5]), [20, 25, 35, 45],
                    interpolation=interpolation,
                )
                self.assertAlmostEqual(result["value_pct"], 100 * math.expm1(0.25), places=9)
                self.assertEqual(result["common_accuracy_interval"], [20.0, 45.0])
                self.assertEqual(result["operating_points"], {"anchor": 5, "test": 4})

    def test_disjoint_or_touching_accuracy_intervals_are_unavailable(self):
        for test_accuracy in ([50, 60, 70, 80], [40, 50, 60, 70]):
            for interpolation in ("pchip", "cubic"):
                with self.subTest(test_accuracy=test_accuracy, interpolation=interpolation):
                    with self.assertRaisesRegex(ValueError, "common accuracy interval"):
                        self.metrics.bd_rate(
                            self.rates, self.accuracy, self.rates, test_accuracy,
                            interpolation=interpolation,
                        )

    def test_nonmonotone_accuracy_cannot_be_repaired_by_sorting_accuracy(self):
        for bad_accuracy in ([10, 30, 20, 40], [40, 30, 20, 10]):
            for side in ("anchor", "test"):
                with self.subTest(accuracy=bad_accuracy, side=side):
                    arguments = [self.rates, self.accuracy, self.rates, self.accuracy]
                    arguments[1 if side == "anchor" else 3] = bad_accuracy
                    with self.assertRaisesRegex(ValueError, "monoton"):
                        self.metrics.bd_rate(*arguments)

    def test_accuracy_plateau_is_excluded_with_an_explicit_reason(self):
        with self.assertRaisesRegex(ValueError, "(?i)(plateau|duplicate accuracy)"):
            self.metrics.bd_rate(
                self.rates, [10, 20, 20, 40], self.rates, self.accuracy
            )

    def test_duplicate_bitrates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "(?i)(duplicate|unique|strictly)"):
            self.metrics.bd_rate(
                [1, 2, 2, 8], self.accuracy, self.rates, self.accuracy
            )

    def test_mismatched_lengths_and_fewer_than_four_points_are_rejected(self):
        invalid_pairs = (
            ([], []), ([1, 2, 3], [10, 20, 30]),
            ([1, 2, 3, 4], [10, 20, 30]),
            ([1, 2, 3], [10, 20, 30, 40]),
        )
        for rate, accuracy in invalid_pairs:
            for side in ("anchor", "test"):
                with self.subTest(rate=rate, accuracy=accuracy, side=side):
                    arguments = [self.rates, self.accuracy, self.rates, self.accuracy]
                    arguments[0 if side == "anchor" else 2] = rate
                    arguments[1 if side == "anchor" else 3] = accuracy
                    with self.assertRaises(ValueError):
                        self.metrics.bd_rate(*arguments)

    def test_nonpositive_bitrates_are_rejected_on_either_curve(self):
        for invalid_rate in (0.0, -1.0):
            for side in (0, 2):
                with self.subTest(invalid_rate=invalid_rate, side=side):
                    arguments = [self.rates, self.accuracy, self.rates, self.accuracy]
                    arguments[side] = [invalid_rate, 2, 4, 8]
                    with self.assertRaisesRegex(ValueError, "positive"):
                        self.metrics.bd_rate(*arguments)

    def test_nonfinite_values_are_rejected_in_every_input_array(self):
        for value in (math.nan, math.inf, -math.inf):
            for index in range(4):
                with self.subTest(value=value, index=index):
                    arguments = [self.rates, self.accuracy, self.rates, self.accuracy]
                    changed = list(arguments[index])
                    changed[1] = value
                    arguments[index] = changed
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.metrics.bd_rate(*arguments)

    def test_zero_ap_is_a_valid_measurement(self):
        result = self.metrics.bd_rate(
            self.rates, [0, 10, 20, 30], self.rates, [0, 10, 20, 30]
        )
        self.assertAlmostEqual(result["value_pct"], 0.0, places=10)

    def test_malformed_arrays_raise_value_error(self):
        for bad_array in (None, 1, "invalid", [[1, 2], [4, 8]], [1, 2, 4, "bad"]):
            with self.subTest(bad_array=bad_array):
                with self.assertRaises(ValueError):
                    self.metrics.bd_rate(
                        bad_array, self.accuracy, self.rates, self.accuracy
                    )

    def test_unsupported_interpolation_is_rejected(self):
        for interpolation in ("linear", "PCHIP", "", None):
            with self.subTest(interpolation=interpolation):
                with self.assertRaisesRegex(ValueError, "interpolation"):
                    self.metrics.bd_rate(
                        self.rates, self.accuracy, self.rates, self.accuracy,
                        interpolation=interpolation,
                    )

    def test_unrepresentable_rate_ratio_does_not_return_infinity(self):
        with self.assertRaisesRegex(ValueError, "(?i)(finite|overflow|representable)"):
            self.metrics.bd_rate(
                [1e-300, 2e-300, 3e-300, 4e-300], self.accuracy,
                [1e300, 2e300, 3e300, 4e300], self.accuracy,
            )


class SensitivityTests(MetricsTestCase):
    def test_pchip_and_cubic_disagreement_matches_hand_integrated_fixture(self):
        # ln(R_anchor)=x; test log-rates are [0,.1,2,3] at x=[0,1,2,3].
        # PCHIP endpoint slopes are 0 and .55. Its integral is the trapezoid
        # sum 3.6 + (0-.55)/12 = 853/240. The cubic 3/8 rule gives 279/80.
        # Subtract anchor integral 9/2 and divide by width 3 to get these
        # independently derived average log ratios.
        expected_pchip = 100 * math.expm1(-227 / 720)
        expected_cubic = 100 * math.expm1(-27 / 80)
        result = self.metrics.compare_bd_rates(
            np.exp([0, 1, 2, 3]), [0, 1, 2, 3],
            np.exp([0, 0.1, 2, 3]), [0, 1, 2, 3],
        )
        self.assertAlmostEqual(result["primary"]["value_pct"], expected_pchip, places=9)
        self.assertAlmostEqual(result["cubic_sensitivity"]["value_pct"], expected_cubic, places=9)
        self.assertEqual(result["primary"]["interpolation"], "pchip")
        self.assertEqual(result["cubic_sensitivity"]["interpolation"], "cubic")
        self.assertAlmostEqual(result["difference_pp"], expected_pchip - expected_cubic, places=9)
        self.assertGreater(abs(result["difference_pp"]), 1.0)
        self.assertIsNone(result["unavailable_reason"])
        json.dumps(result, allow_nan=False)

    def test_invalid_comparisons_return_null_metrics_and_explain_why(self):
        cases = (
            ([10, 20, 20, 40], self.accuracy, "(?i)(plateau|duplicate accuracy)"),
            ([10, 30, 20, 40], self.accuracy, "monoton"),
            (self.accuracy, [50, 60, 70, 80], "common accuracy interval"),
        )
        for anchor_accuracy, test_accuracy, reason in cases:
            with self.subTest(anchor_accuracy=anchor_accuracy, test_accuracy=test_accuracy):
                result = self.metrics.compare_bd_rates(
                    self.rates, anchor_accuracy, self.rates, test_accuracy
                )
                self.assertIsNone(result["primary"])
                self.assertIsNone(result["cubic_sensitivity"])
                self.assertIsNone(result["difference_pp"])
                self.assertRegex(result["unavailable_reason"], reason)
                json.dumps(result, allow_nan=False)

    def test_overflow_comparison_remains_json_serializable(self):
        result = self.metrics.compare_bd_rates(
            [1e-300, 2e-300, 3e-300, 4e-300], self.accuracy,
            [1e300, 2e300, 3e300, 4e300], self.accuracy,
        )
        self.assertIsNone(result["primary"])
        self.assertIsNone(result["cubic_sensitivity"])
        self.assertIsNone(result["difference_pp"])
        self.assertTrue(result["unavailable_reason"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
