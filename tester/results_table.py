"""results_table.py ->Tabular results management for FJSP solver comparisons.

Stores instance characterization metrics, optimal algorithm selections, makespans,
runtimes, and optimality error gaps (BKS gap) in a tabular format (CSV and Markdown).
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "my_instance_results.csv"

# Canonical column order for the tabular results
FIELDNAMES = [
    "timestamp",
    "instance_name",
    "n_jobs",
    "n_machines",
    "n_operations",
    "avg_ops_per_job",
    "beta",
    "dv",
    "eligibility_entropy",
    "processing_gap_ratio",
    "time_skew",
    "bottleneck_metric",
    "bottleneck_coverage",
    "bottleneck_intensity",
    "optimal_algorithm",
    "selection_reason",
    "status",
    "makespan",
    "runtime_seconds",
    "bks_lower_bound",
    "bks_upper_bound",
    "bks_gap_pct",
    "instance_path",
]


def build_result_row(
    instance_summary: Dict[str, Any],
    result: Dict[str, Any],
    optimal_algorithm: str,
    selection_reason: str,
    instance_name: str,
    instance_path: Union[str, Path],
    gap_info: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized dictionary row for tabular storage."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    path_str = str(instance_path)
    try:
        rel_path = str(Path(instance_path).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, Exception):
        rel_path = path_str

    validation = result.get("validation", {})
    is_valid = validation.get("is_valid", True) if isinstance(validation, dict) else True
    status = "VALID" if is_valid else "INVALID"

    bks_lb = gap_info.get("lower_bound") if gap_info else ""
    bks_ub = gap_info.get("upper_bound") if gap_info else ""
    bks_gap = round(gap_info["gap_pct"], 2) if gap_info else ""

    return {
        "timestamp": timestamp,
        "instance_name": instance_name,
        "n_jobs": int(instance_summary.get("n_jobs", 0)),
        "n_machines": int(instance_summary.get("n_machines", 0)),
        "n_operations": int(instance_summary.get("n_operations", 0)),
        "avg_ops_per_job": round(float(instance_summary.get("avg_ops_per_job", 0.0)), 2),
        "beta": round(float(instance_summary.get("beta", 0.0)), 3),
        "dv": round(float(instance_summary.get("dv", 0.0)), 3),
        "eligibility_entropy": round(float(instance_summary.get("machine_eligibility_entropy", 0.0)), 3),
        "processing_gap_ratio": round(float(instance_summary.get("processing_time_gap_ratio", 0.0)), 3),
        "time_skew": round(float(instance_summary.get("time_skew", 0.0)), 3),
        "bottleneck_metric": round(float(instance_summary.get("bottleneck_metric", 0.0)), 3),
        "bottleneck_coverage": round(float(instance_summary.get("bottleneck_coverage", 0.0)), 3),
        "bottleneck_intensity": round(float(instance_summary.get("bottleneck_intensity", 0.0)), 3),
        "optimal_algorithm": optimal_algorithm,
        "selection_reason": selection_reason,
        "status": status,
        "makespan": int(result.get("makespan", 0)),
        "runtime_seconds": round(float(result.get("runtime_seconds", 0.0)), 2),
        "bks_lower_bound": bks_lb,
        "bks_upper_bound": bks_ub,
        "bks_gap_pct": bks_gap,
        "instance_path": rel_path,
    }


def _migrate_legacy_results_csv(csv_file: Path) -> None:
    """Keep existing result rows aligned when the BKS columns change."""
    with csv_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        if old_fieldnames == FIELDNAMES:
            return
        if not {
            "instance_name",
            "bks_lower_bound",
            "bks_upper_bound",
            "bks_gap_pct",
            "bks_gap_to_lb_pct",
        }.issubset(old_fieldnames):
            raise ValueError(
                f"Cannot migrate results CSV with unexpected columns: {old_fieldnames}"
            )
        old_rows = list(reader)

    migrated_rows = []
    for old_row in old_rows:
        row = {field: old_row.get(field, "") for field in FIELDNAMES}
        if not row["bottleneck_metric"]:
            try:
                row["bottleneck_metric"] = round(
                    0.5 * float(old_row["bottleneck_coverage"])
                    + 0.5 * float(old_row["bottleneck_intensity"]),
                    3,
                )
            except (KeyError, TypeError, ValueError):
                row["bottleneck_metric"] = ""

        gaps = [
            abs(float(old_row[key]))
            for key in ("bks_gap_pct", "bks_gap_to_lb_pct")
            if old_row.get(key) not in (None, "")
        ]
        row["bks_gap_pct"] = round(min(gaps), 2) if gaps else ""
        migrated_rows.append(row)

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(migrated_rows)


def record_result(
    instance_summary: Dict[str, Any],
    result: Dict[str, Any],
    optimal_algorithm: str,
    selection_reason: str,
    instance_name: str,
    instance_path: Union[str, Path],
    gap_info: Optional[Dict[str, Any]] = None,
    csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV,
) -> Dict[str, Any]:
    """Append a benchmark comparison run to the tabular CSV file."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    row = build_result_row(
        instance_summary=instance_summary,
        result=result,
        optimal_algorithm=optimal_algorithm,
        selection_reason=selection_reason,
        instance_name=instance_name,
        instance_path=instance_path,
        gap_info=gap_info,
    )

    file_exists = csv_file.exists() and csv_file.stat().st_size > 0
    if file_exists:
        _migrate_legacy_results_csv(csv_file)
    with csv_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def load_results(csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV) -> List[Dict[str, str]]:
    """Load all records from the results CSV table."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_results_table(
    csv_path: Union[str, Path] = DEFAULT_RESULTS_CSV,
    last_n: Optional[int] = None,
    filter_instance: Optional[str] = None,
) -> None:
    """Print an ASCII table of stored results to the console."""
    rows = load_results(csv_path)
    if filter_instance:
        rows = [r for r in rows if filter_instance.lower() in r.get("instance_name", "").lower()]
    if last_n is not None and last_n > 0:
        rows = rows[-last_n:]

    if not rows:
        print(f"No results found in {csv_path}")
        return

    header = (
        f"{'instance':<15} {'jobs':>4} {'mach':>4} {'ops':>4} {'beta':>6} "
        f"{'dv':>6} {'ent':>6} {'gap_r':>6} {'skew':>6} {'diff':>6} "
        f"{'algorithm':<22} {'makespan':>8} {'runtime':>7} {'LB':>6} {'UB':>6} {'BKS gap':>9}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in rows:
        gap_value = r.get("bks_gap_pct")
        gap_display = f"{float(gap_value):.1f}%" if gap_value not in (None, "") else "n/a"
        lower_bound = r.get("bks_lower_bound") or "n/a"
        upper_bound = r.get("bks_upper_bound") or "n/a"
        algo_display = r.get("optimal_algorithm", "")[:22]
        print(
            f"{r.get('instance_name', '')[:15]:<15} "
            f"{int(r.get('n_jobs', 0)):4d} "
            f"{int(r.get('n_machines', 0)):4d} "
            f"{int(r.get('n_operations', 0)):4d} "
            f"{float(r.get('beta', 0)):6.3f} "
            f"{float(r.get('dv', 0)):6.3f} "
            f"{float(r.get('eligibility_entropy', 0)):6.3f} "
            f"{float(r.get('processing_gap_ratio', 0)):6.3f} "
            f"{float(r.get('time_skew', 0)):6.3f} "
            f"{float(r.get('bottleneck_metric', 0)):6.3f} "
            f"{algo_display:<22} "
            f"{int(r.get('makespan', 0)):8d} "
            f"{float(r.get('runtime_seconds', 0)):6.2f}s "
            f"{lower_bound:>6} {upper_bound:>6} "
            f"{gap_display:>12}"
        )
    print("=" * len(header))
