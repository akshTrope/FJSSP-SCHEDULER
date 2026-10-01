"""Best-known lookup utilities for the benchmark instances shipped with this repo."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _normalise_instance_name(name: str) -> str:
    if name is None:
        return ""
    return name.lower().replace("_", "").replace(".txt", "").strip()


def _load_bks() -> list[Dict[str, Any]]:
    bks_path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "flexible-jobshop"
        / "solutions"
        / "bks.json"
    )
    with bks_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_BKS_CACHE = _load_bks()


def get_best_known(instance_name: str) -> Optional[Tuple[int, int]]:
    """Return (lower_bound, upper_bound) for a benchmark instance, if present."""
    target = _normalise_instance_name(instance_name)
    for entry in _BKS_CACHE:
        if _normalise_instance_name(entry.get("instance", "")) == target:
            lower_bound = int(entry.get("lower_bound", 0))
            upper_bound = int(entry.get("upper_bound", lower_bound))
            return lower_bound, upper_bound
    return None


def compute_gap(achieved_makespan: int, instance_name: str) -> Optional[dict]:
    """Compute the smaller percentage distance to the published bounds."""
    bounds = get_best_known(instance_name)
    if bounds is None:
        return None

    lower_bound, upper_bound = bounds
    if upper_bound == 0:
        return None

    gap_to_upper_bound_pct = 100.0 * (achieved_makespan - upper_bound) / upper_bound
    gap_to_lower_bound_pct = 100.0 * (achieved_makespan - lower_bound) / lower_bound

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "is_proven_optimal": lower_bound == upper_bound,
        "gap_pct": min(abs(gap_to_lower_bound_pct), abs(gap_to_upper_bound_pct)),
    }
