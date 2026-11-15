"""Matched v3 experiment identities, separate from preserved v2 artifacts.

The first four variants form a temporal-package by semantic-QG 2x2 factorial.
The proxy control changes only codec training. The paper-derived image reference
is a separate adaptation, not an official SCOPE reproduction. Each job uses the
entire common training split; accounts are execution slots, never data shards.
"""
from typing import Any, Dict, Iterable, List


VARIANTS = (
    "spatial",
    "spatial_qp",
    "temporal",
    "temporal_qp",
    "temporal_qp_proxy",
    "scope_paper_adapted",
)
SEEDS = (2026, 2027, 2028)


def variant_settings(name: str) -> Dict[str, Any]:
    """Return a fresh settings dictionary for an exact, versioned variant name."""
    if not isinstance(name, str) or name not in VARIANTS:
        raise ValueError(f"Unknown v3 variant: {name!r}")
    temporal = name in ("temporal", "temporal_qp", "temporal_qp_proxy")
    qp = name in ("spatial_qp", "temporal_qp", "temporal_qp_proxy")
    return {
        "disable_temporal": not temporal,
        "max_delta_qp": 4.0 if qp else 0.0,
        "use_real_x265_ste": name not in ("temporal_qp_proxy", "scope_paper_adapted"),
        "lambda_temp_bg": 1.0 if temporal else 0.0,
        "lambda_temp_obj": 0.5 if temporal else 0.0,
        "model_kind": "scope_paper_adapted" if name == "scope_paper_adapted" else "temporal_scope",
    }


def build_model(name: str, **kwargs: Any):
    """Build a variant without reseeding or changing the matched backbone.

    Callers seed immediately before construction. Temporal disabling is a forward
    flag, so the four factorial cells retain identical parameter initialization.
    Architecture overrides support small offline tests; the variant's QG scale
    cannot be overridden to silently change the experimental intervention.
    """
    settings = variant_settings(name)
    scale = settings["max_delta_qp"]
    if "max_delta_qp" in kwargs and kwargs["max_delta_qp"] != scale:
        raise ValueError(f"max_delta_qp conflicts with v3 variant {name!r}")
    kwargs["max_delta_qp"] = scale
    if settings["model_kind"] == "scope_paper_adapted":
        from models.scope_reference import ScopePaperAdapted

        return ScopePaperAdapted(**kwargs)
    from models.temporal_scope import TemporalSCOPE

    return TemporalSCOPE(**kwargs)


def build_job_matrix(seeds: Iterable[int] = SEEDS) -> List[Dict[str, Any]]:
    """Return stable variant-major complete replicates with unique job IDs.

    Seed values are explicit unsigned 32-bit Python integers so Python, NumPy
    and PyTorch can share the same identity without coercion or modulo changes.
    """
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a nonempty iterable of integer seeds")
    try:
        values = tuple(seeds)
    except TypeError as error:
        raise ValueError("seeds must be a nonempty iterable of integer seeds") from error
    if not values or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
        raise ValueError("seeds must contain unsigned 32-bit Python integers, excluding bool")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique to preserve distinct job identities")
    return [
        {
            "job_id": f"{variant}_seed{seed}",
            "variant": variant,
            "seed": seed,
            "settings": variant_settings(variant),
        }
        for variant in VARIANTS
        for seed in values
    ]
