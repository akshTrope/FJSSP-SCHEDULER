"""
Save/load generated instances to/from JSON.

JSON object keys must be strings, but our instance dict uses tuple keys
((j,k) for E, (j,k,m) for P) -- these are encoded as "j,k" / "j,k,m"
strings on save and decoded back into int tuples on load.
"""

import json
from pathlib import Path
from typing import Tuple


def _encode_instance(instance: dict) -> dict:
    return {
        "jobs": {str(j): ops for j, ops in instance["jobs"].items()},
        "machines": instance["machines"],
        "E": {f"{j},{k}": eligible for (j, k), eligible in instance["E"].items()},
        "P": {
            f"{j},{k},{m}": p_val
            for (j, k, m), p_val in instance["P"].items()
        },
    }


def _decode_instance(raw: dict) -> dict:
    jobs = {int(j): ops for j, ops in raw["jobs"].items()}

    E = {}
    for key, eligible in raw["E"].items():
        j_str, k_str = key.split(",")
        E[(int(j_str), int(k_str))] = eligible

    P = {}
    for key, p_val in raw["P"].items():
        j_str, k_str, m_str = key.split(",")
        P[(int(j_str), int(k_str), int(m_str))] = p_val

    return {
        "jobs": jobs,
        "machines": raw["machines"],
        "E": E,
        "P": P,
    }


def save_instance(instance: dict, metadata: dict, path: str) -> None:
    """Writes {"instance": ..., "metadata": ...} as a single JSON file."""
    payload = {
        "instance": _encode_instance(instance),
        "metadata": metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_instance(path: str) -> Tuple[dict, dict]:
    """Reads back a file written by save_instance(). Returns (instance, metadata)."""
    with open(path, "r") as f:
        payload = json.load(f)
    instance = _decode_instance(payload["instance"])
    metadata = payload["metadata"]
    return instance, metadata
