"""Strict v3 Bjontegaard rate/AP comparison, independent of historical v2.

The primary metric integrates natural-log bitrate with shape-preserving PCHIP.
Cubic fitting is a sensitivity analysis. Neither method extrapolates beyond the
common accuracy interval. A negative percentage means the test saves bitrate.

AP plateaus are excluded: duplicate AP values do not define a unique inverse
rate-at-accuracy curve. They are reported as unavailable rather than merged,
jittered, or removed. Nonmonotone curves are likewise never silently repaired.
"""

import math
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator


def _curve(rate, accuracy, label):
    """Validate paired operating points and sort only by measured bitrate."""
    arrays = []
    for values, name in ((rate, "bitrate"), (accuracy, "accuracy")):
        try:
            if np.iscomplexobj(values):
                raise ValueError("complex measurements are not supported")
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} {name} must be a finite numeric array") from error
        if array.ndim != 1:
            raise ValueError(f"{label} {name} must be a one-dimensional array")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} {name} measurements must be finite")
        arrays.append(array)
    rate, accuracy = arrays
    if len(rate) != len(accuracy):
        raise ValueError(f"{label} bitrate and accuracy lengths must match")
    if len(rate) < 4:
        raise ValueError(f"{label} requires at least four operating points")
    if np.any(rate <= 0):
        raise ValueError(f"{label} bitrates must be strictly positive")
    order = np.argsort(rate)
    rate, accuracy = rate[order], accuracy[order]
    if np.any(rate[1:] == rate[:-1]):
        raise ValueError(f"{label} has duplicate bitrates; unique operating points are required")
    if len(np.unique(accuracy)) != len(accuracy):
        raise ValueError(
            f"{label} has duplicate accuracy values (AP plateau); "
            "rate-at-accuracy is not unique, so the curve is excluded"
        )
    if np.any(accuracy[1:] <= accuracy[:-1]):
        raise ValueError(f"{label} accuracy must be strictly monotone increasing with bitrate")
    return rate, accuracy


def _mean_log_rate(rate, accuracy, lo, hi, interpolation):
    """Integrate the measured inverse curve over an already validated overlap."""
    log_rate = np.log(rate)
    if interpolation == "pchip":
        curve = PchipInterpolator(accuracy, log_rate, extrapolate=False)
        return float(curve.integrate(lo, hi, extrapolate=False) / (hi - lo))
    # An affine change of AP coordinates improves conditioning without changing
    # the fitted cubic or its mean over the common interval [0, 1]. All measured
    # operating points constrain the fit; only the overlap is integrated.
    normalized_accuracy = (accuracy - lo) / (hi - lo)
    coefficients = np.polyfit(normalized_accuracy, log_rate, 3)
    integral = np.polyint(coefficients)
    return float(np.polyval(integral, 1.0) - np.polyval(integral, 0.0))


def bd_rate(rate_anchor, acc_anchor, rate_test, acc_test, interpolation="pchip"):
    """Return BD-rate percent and diagnostics for two measured rate/AP curves.

    Inputs are paired one-dimensional arrays with at least four points each.
    Bitrates must be positive; accuracies must be finite and strictly increase
    with bitrate. Input order may be arbitrary, including descending CRF rate
    order. Zero AP is a valid measurement. The returned ``operating_points``
    counts include every supplied point used to construct each curve.

    ``interpolation='pchip'`` is primary; ``'cubic'`` uses a least-squares cubic
    of natural-log bitrate for sensitivity analysis. Both integrate only the
    positive-width intersection of measured accuracy ranges. Invalid inputs,
    AP plateaus, nonmonotone curves and unavailable/nonfinite results raise
    ``ValueError``. Successful results contain only finite JSON-native values.
    """
    if interpolation not in ("pchip", "cubic"):
        raise ValueError("interpolation must be 'pchip' or 'cubic'")
    rate_anchor, acc_anchor = _curve(rate_anchor, acc_anchor, "anchor")
    rate_test, acc_test = _curve(rate_test, acc_test, "test")
    lo = float(max(acc_anchor[0], acc_test[0]))
    hi = float(min(acc_anchor[-1], acc_test[-1]))
    if hi <= lo:
        raise ValueError("No positive common accuracy interval; extrapolation is prohibited")
    try:
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise"):
            warnings.simplefilter("error")
            if not math.isfinite(hi - lo):
                raise ValueError("Common accuracy interval width must be finite")
            anchor_mean = _mean_log_rate(rate_anchor, acc_anchor, lo, hi, interpolation)
            test_mean = _mean_log_rate(rate_test, acc_test, lo, hi, interpolation)
            value = 100.0 * math.expm1(test_mean - anchor_mean)
    except (ArithmeticError, np.linalg.LinAlgError, Warning) as error:
        raise ValueError(f"{interpolation} result is not finite or numerically representable") from error
    if not math.isfinite(value):
        raise ValueError(f"{interpolation} result is not finite or numerically representable")
    return {
        "value_pct": float(value),
        "common_accuracy_interval": [lo, hi],
        "interpolation": interpolation,
        "operating_points": {"anchor": len(rate_anchor), "test": len(rate_test)},
    }


def compare_bd_rates(rate_anchor, acc_anchor, rate_test, acc_test):
    """Compare primary PCHIP with cubic sensitivity without inventing results.

    ``difference_pp`` is the signed PCHIP percentage minus the cubic percentage,
    in percentage points. An unavailable method returns ``None`` and an explicit
    ``unavailable_reason``; the difference is available only when both succeed.
    """
    results = {}
    reasons = []
    for field, interpolation in (("primary", "pchip"), ("cubic_sensitivity", "cubic")):
        try:
            results[field] = bd_rate(
                rate_anchor, acc_anchor, rate_test, acc_test,
                interpolation=interpolation,
            )
        except ValueError as error:
            results[field] = None
            reasons.append(f"{interpolation}: {error}")
    results["difference_pp"] = (
        results["primary"]["value_pct"] - results["cubic_sensitivity"]["value_pct"]
        if all(result is not None for result in results.values()) else None
    )
    results["unavailable_reason"] = "; ".join(reasons) if reasons else None
    return results
