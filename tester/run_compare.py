"""Select the final FJSP solution using CP-SAT first, then the WOA+LNS+CP-SAT hybrid.

Change INSTANCE_NAME below, or pass via command line:
    python3 -m tester.run_compare
    python3 -m tester.run_compare behnke27
    python3 -m tester.run_compare --path some_instance.json
    python3 -m tester.run_compare --path path/to/a/directory/of/instances
    python3 -m tester.run_compare --view-table
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from algorithms.cp_sat import solve_cp_sat
from algorithms.encoding import schedule_to_entries
from algorithms.solve import solve_fjsp
from algorithms.validator_fn import InstanceValidationError, validate_schedule
from generator.metrics import summarize_instance
from generator.io_utils import load_instance
from tester.best_known_results import compute_gap
from tester.fjsplib_parser import parse_fjsplib_file
from tester.results_table import (
    DEFAULT_RESULTS_CSV,
    print_results_table,
    record_result,
)


# For a standard benchmark, leave INSTANCE_PATH as None and change INSTANCE_NAME.
# For a generated JSON or explicit FJSPLib text instance, set INSTANCE_PATH to
# either a single file OR a directory of files -- both are handled below.
INSTANCE_NAME = "behnke1"
INSTANCE_PATH: Optional[str] = None

# Tabular results logging settings
SAVE_TO_TABLE = False
TABLE_FILE_PATH = DEFAULT_RESULTS_CSV

# CP-SAT is always attempted first with this fixed budget.
CP_SAT_MAX_TIME_SECONDS = 10
CP_SAT_NUM_SEARCH_WORKERS = 8
CP_SAT_LOG_SEARCH_PROGRESS = False

# WOA + LNS hybrid settings. Each seed gets its own full time budget.
HYBRID_SEEDS = (0, 1, 2, 3)
HYBRID_MAX_TIME_SECONDS = 15
WOA_POPULATION_SIZE = 30
WOA_MAX_ITERATIONS = 100
LNS_FREQUENCY = 5
LNS_WINDOW_SIZE = 10
LNS_CP_SAT_MAX_TIME_SECONDS = 3.0

# Set to True when you want every operation printed for best schedule.
PRINT_SCHEDULE = False


def find_benchmark(instance_name: str) -> Path:
    """Find a standard FJSPLib benchmark file by filename stem."""
    project_root = Path(__file__).resolve().parent.parent
    benchmark_root = project_root / "benchmarks" / "flexible-jobshop" / "instances" / "fjsp"
    matches = sorted(benchmark_root.rglob(f"{instance_name}.txt"))

    if not matches:
        raise FileNotFoundError(
            f"No benchmark named '{instance_name}' was found under {benchmark_root}."
        )
    if len(matches) > 1:
        paths = ", ".join(str(path.relative_to(project_root)) for path in matches)
        raise RuntimeError(f"Benchmark name '{instance_name}' is ambiguous: {paths}")
    return matches[0]


def load_single_file(instance_path: Path) -> Tuple[Path, str, Dict[str, Any], Any]:
    """Load exactly one instance file (.json or .txt) and return its parsed form."""
    if instance_path.suffix.lower() == ".json":
        instance, metadata = load_instance(str(instance_path))
    elif instance_path.suffix.lower() == ".txt":
        instance = parse_fjsplib_file(instance_path)
        metadata = None
    else:
        raise ValueError(
            f"Unsupported instance file type '{instance_path.suffix}'. "
            "Use a generated .json or FJSPLib .txt file."
        )
    return instance_path, instance_name_from_path(instance_path), instance, metadata


def load_test_instance(
    custom_name: Optional[str] = None,
    custom_path: Optional[str] = None,
) -> Tuple[Path, str, Dict[str, Any], Any]:
    """
    Load exactly ONE instance:
      - if a path is given (custom_path or the module-level INSTANCE_PATH) and
        it points to a single FILE, load that file directly.
      - otherwise, treat the name (custom_name or the module-level
        INSTANCE_NAME) as a standard benchmark stem to look up.

    Does NOT handle directories -- that's the batch path in main().
    """
    project_root = Path(__file__).resolve().parent.parent
    target_path = custom_path if custom_path is not None else INSTANCE_PATH
    target_name = custom_name if custom_name is not None else INSTANCE_NAME

    if target_path is not None:
        instance_path = Path(target_path)
        if not instance_path.is_absolute():
            instance_path = project_root / instance_path
        if instance_path.is_dir():
            raise IsADirectoryError(
                f"'{instance_path}' is a directory, not a file -- use the batch "
                "runner (main() with a directory path) instead of load_test_instance()."
            )
        return load_single_file(instance_path)

    benchmark_path = find_benchmark(target_name)
    return load_single_file(benchmark_path)


def instance_name_from_path(instance_path: Path) -> str:
    """Return the BKS identity, including Hurink's variant folder."""
    variant = instance_path.parent.name.lower()
    if variant in {"edata", "rdata", "sdata", "vdata"}:
        return f"{instance_path.stem}_{variant}"
    return instance_path.stem


def validate_cp_sat_result(instance: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate CP-SAT's schedule and return a compact result summary."""
    try:
        validation = validate_schedule(instance, result["schedule"])
        return {
            "is_valid": True,
            "validated_makespan": validation["makespan"],
        }
    except InstanceValidationError as error:
        return {
            "is_valid": False,
            "validated_makespan": None,
            "validation_error": str(error),
        }


def validate_hybrid_result(instance: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the solve.py hybrid result using the shared validator."""
    try:
        validation = validate_schedule(instance, schedule_to_entries(result["schedule_by_op"]))
        return {
            "is_valid": True,
            "validated_makespan": validation["makespan"],
        }
    except InstanceValidationError as error:
        return {
            "is_valid": False,
            "validated_makespan": None,
            "validation_error": str(error),
        }


def print_gap(gap_info: Any) -> None:
    if gap_info is None:
        print("BKS gap:     unavailable")
        return
    print(
        f"BKS gap:     {gap_info['gap_pct']:.1f}% "
        f"(LB={gap_info['lower_bound']}, UB={gap_info['upper_bound']})"
    )


def print_instance_metrics(summary: Dict[str, Any]) -> None:
    """Print structural metrics shared by both solver results."""
    print("=== Instance Metrics ===")
    print(
        f"size:      jobs={summary['n_jobs']} machines={summary['n_machines']} "
        f"operations={summary['n_operations']}"
    )
    print(
        f"beta={summary['beta']:.3f}  dv={summary['dv']:.3f}  "
        f"eligibility_entropy={summary['machine_eligibility_entropy']:.3f}"
    )
    print(
        f"processing_gap={summary['processing_time_gap_ratio']:.3f}  "
        f"time_skew={summary['time_skew']:.3f}"
    )
    print(
        f"bottleneck_coverage={summary['bottleneck_coverage']:.3f}  "
        f"bottleneck_intensity={summary['bottleneck_intensity']:.3f}  "
        f"bottleneck_metric={summary['bottleneck_metric']:.3f}"
    )
    if "congestion_ratio" in summary:
        print(f"congestion_ratio={summary['congestion_ratio']:.3f}")


def print_schedule(title: str, schedule: list) -> None:
    """Print operations in machine execution order."""
    print(f"--- {title} ---")
    print(" job  op  machine  start  finish")
    print(" ---  --  -------  -----  ------")
    ordered_schedule = sorted(schedule, key=lambda row: (row[3], row[0], row[1], row[2]))
    for job_id, operation_id, machine_id, start_time, finish_time in ordered_schedule:
        print(
            f" {job_id:>3}  {operation_id:>2}  {machine_id:>7}  "
            f"{start_time:>5}  {finish_time:>6}"
        )


def run_hybrid(
    instance: Dict[str, Any],
    instance_name: str,
    initial_cpsat_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run four seeded WOA+LNS trials, reusing the initial CP-SAT result."""
    started_at = time.perf_counter()
    trials = []
    for seed in HYBRID_SEEDS:
        trial_started_at = time.perf_counter()
        result = solve_fjsp(
            instance,
            hybrid_time_budget=HYBRID_MAX_TIME_SECONDS,
            woa_population_size=WOA_POPULATION_SIZE,
            woa_max_iterations=WOA_MAX_ITERATIONS,
            lns_frequency=LNS_FREQUENCY,
            lns_window_size=LNS_WINDOW_SIZE,
            lns_time_limit=LNS_CP_SAT_MAX_TIME_SECONDS,
            seed=seed,
            initial_cpsat_result=initial_cpsat_result,
            skip_cpsat=True,
        )
        trials.append({
            **result,
            "seed": seed,
            "runtime_seconds": time.perf_counter() - trial_started_at,
            "validation": validate_hybrid_result(instance, result),
        })

    valid_trials = [trial for trial in trials if trial["validation"]["is_valid"]]
    best_trial = min(valid_trials or trials, key=lambda trial: trial["makespan"])
    return {
        **best_trial,
        "runtime_seconds": time.perf_counter() - started_at,
        "instance_name": instance_name,
        "best_seed": best_trial["seed"],
        "seed_results": [
            {
                "seed": trial["seed"],
                "makespan": trial["makespan"],
                "runtime_seconds": trial["runtime_seconds"],
                "is_valid": trial["validation"]["is_valid"],
            }
            for trial in trials
        ],
    }


def print_final_result(
    instance_path: Path,
    instance_name: str,
    instance_summary: Dict[str, Any],
    result: Dict[str, Any],
    reason: str,
    optimal_algorithm: str = "Unknown",
    save_to_table: bool = True,
    table_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Print only the selected final solution and save to the tabular results file."""
    project_root = Path(__file__).resolve().parent.parent
    validation = result["validation"]
    gap_info = compute_gap(result["makespan"], instance_name)

    try:
        rel_instance_path = instance_path.relative_to(project_root)
    except (ValueError, Exception):
        rel_instance_path = instance_path

    print(f"Instance:    {rel_instance_path}")
    print_instance_metrics(instance_summary)
    print()
    print("=== Selected Final Solution ===")
    print(f"optimal alg: {optimal_algorithm}")
    print(f"selected by: {reason}")
    print(f"status:      {'VALID' if validation['is_valid'] else 'INVALID'}")
    print(f"makespan:    {result['makespan']}")
    print(f"runtime:     {result['runtime_seconds']:.2f}s")
    if "seed_results" in result:
        trial_summary = ", ".join(
            f"{trial['seed']}:{trial['makespan']}"
            for trial in result["seed_results"]
        )
        print(
            f"hybrid seeds (seed:makespan): {trial_summary}; "
            f"best seed={result['best_seed']}"
        )
    print_gap(gap_info)
    if not validation["is_valid"]:
        print(f"validation error: {validation['validation_error']}")
    if PRINT_SCHEDULE:
        print_schedule("Selected schedule", result["schedule"])

    row = None
    if save_to_table:
        target_csv = table_path or TABLE_FILE_PATH
        row = record_result(
            instance_summary=instance_summary,
            result=result,
            optimal_algorithm=optimal_algorithm,
            selection_reason=reason,
            instance_name=instance_name,
            instance_path=instance_path,
            gap_info=gap_info,
            csv_path=target_csv,
        )
        try:
            rel_csv = Path(target_csv).resolve().relative_to(project_root)
        except (ValueError, Exception):
            rel_csv = target_csv
        print(f"\n[Tabular Entry Recorded] -> {rel_csv}")

    return row


def compare_instance(
    instance_path: Path,
    instance_name: str,
    instance: Dict[str, Any],
    metadata: Any = None,
    save_to_table: bool = True,
    table_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the CP-SAT vs Hybrid comparison and store the best result."""
    params = metadata.get("params", {}) if metadata else {}
    bottleneck_machines = params.get("bottleneck_machines")
    instance_summary = summarize_instance(
        instance,
        bottleneck_machines=bottleneck_machines,
    )

    comparison_started_at = time.perf_counter()
    try:
        cp_sat_result = solve_cp_sat(
            instance,
            max_time_in_seconds=CP_SAT_MAX_TIME_SECONDS,
            num_search_workers=CP_SAT_NUM_SEARCH_WORKERS,
            log_search_progress=CP_SAT_LOG_SEARCH_PROGRESS,
        )
        cp_sat_validation = validate_cp_sat_result(instance, cp_sat_result)

        if cp_sat_result["status"] == "OPTIMAL" and cp_sat_validation["is_valid"]:
            selected_result = {
                "makespan": cp_sat_result["makespan"],
                "runtime_seconds": cp_sat_result["time_seconds"],
                "schedule": cp_sat_result["schedule"],
                "validation": cp_sat_validation,
            }
            optimal_algorithm = "CP-SAT (OPTIMAL)"
            reason = "CP-SAT OPTIMAL and VALID"
            selected_result["runtime_seconds"] = time.perf_counter() - comparison_started_at
            print_final_result(
                instance_path,
                instance_name,
                instance_summary,
                selected_result,
                reason,
                optimal_algorithm=optimal_algorithm,
                save_to_table=save_to_table,
                table_path=table_path,
            )
            return selected_result

        if cp_sat_result["status"] == "FEASIBLE" and cp_sat_validation["is_valid"]:
            hybrid_result = run_hybrid(
                instance,
                instance_name,
                initial_cpsat_result=cp_sat_result,
            )
            if hybrid_result["validation"]["is_valid"] and hybrid_result["makespan"] < cp_sat_result["makespan"]:
                selected_result = {
                    "makespan": hybrid_result["makespan"],
                    "runtime_seconds": hybrid_result["runtime_seconds"],
                    "schedule": schedule_to_entries(hybrid_result["schedule_by_op"]),
                    "validation": hybrid_result["validation"],
                }
                optimal_algorithm = "Hybrid (best of 4 seeds)"
                reason = f"Best hybrid seed {hybrid_result['best_seed']} beat CP-SAT FEASIBLE"
            else:
                selected_result = {
                    "makespan": cp_sat_result["makespan"],
                    "runtime_seconds": cp_sat_result["time_seconds"],
                    "schedule": cp_sat_result["schedule"],
                    "validation": cp_sat_validation,
                }
                optimal_algorithm = "CP-SAT (FEASIBLE)"
                reason = f"CP-SAT FEASIBLE no worse than best hybrid seed {hybrid_result['best_seed']}"
            selected_result["seed_results"] = hybrid_result["seed_results"]
            selected_result["best_seed"] = hybrid_result["best_seed"]
            selected_result["runtime_seconds"] = time.perf_counter() - comparison_started_at
            print_final_result(
                instance_path,
                instance_name,
                instance_summary,
                selected_result,
                reason,
                optimal_algorithm=optimal_algorithm,
                save_to_table=save_to_table,
                table_path=table_path,
            )
            return selected_result

        reason = f"CP-SAT returned {cp_sat_result['status']} or an invalid schedule"
    except RuntimeError as error:
        if "CP-SAT did not find a feasible schedule" not in str(error):
            raise
        reason = "Hybrid fallback after CP-SAT found no feasible schedule"

    hybrid_result = run_hybrid(instance, instance_name)
    selected_result = {
        "makespan": hybrid_result["makespan"],
        "runtime_seconds": hybrid_result["runtime_seconds"],
        "schedule": schedule_to_entries(hybrid_result["schedule_by_op"]),
        "validation": hybrid_result["validation"],
        "seed_results": hybrid_result["seed_results"],
        "best_seed": hybrid_result["best_seed"],
    }
    optimal_algorithm = "Hybrid (best of 4 seeds)"
    reason = f"{reason}; best hybrid seed {hybrid_result['best_seed']}"
    selected_result["runtime_seconds"] = time.perf_counter() - comparison_started_at
    print_final_result(
        instance_path,
        instance_name,
        instance_summary,
        selected_result,
        reason,
        optimal_algorithm=optimal_algorithm,
        save_to_table=save_to_table,
        table_path=table_path,
    )
    return selected_result


def run_directory(instance_dir: Path, save_to_table: bool, table_path: Path) -> None:
    """Run every supported instance file inside a directory."""
    instance_files = sorted(
        path for path in instance_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".json"}
    )

    if not instance_files:
        raise FileNotFoundError(
            f"No .txt or .json instance files found in {instance_dir}"
        )

    for instance_file in instance_files:
        print(f"\n{'=' * 20} Running {instance_file.name} {'=' * 20}")

        inst_path, inst_name, inst, meta = load_single_file(instance_file)

        compare_instance(
            instance_path=inst_path,
            instance_name=inst_name,
            instance=inst,
            metadata=meta,
            save_to_table=save_to_table,
            table_path=table_path,
        )

    print("\nAll files in directory completed.")
    print_results_table(csv_path=table_path, last_n=len(instance_files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select final FJSP solution using CP-SAT and the WOA+LNS hybrid, "
        "and store instance metrics + selected algorithm + BKS gap into a tabular file."
    )
    parser.add_argument(
        "instance_name",
        nargs="?",
        default=None,
        help="Optional benchmark instance name (e.g. behnke27, mk01, brandimarte_mk01)",
    )
    parser.add_argument(
        "--path",
        "-p",
        default=None,
        help="Optional path to a SINGLE instance file (.txt or .json), or to a "
        "DIRECTORY of instance files to run in sequence.",
    )
    parser.add_argument(
        "--table",
        "-t",
        default=None,
        help="Custom path to output CSV table (default: results/my_instance_results.csv)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable automatic saving to tabular file",
    )
    parser.add_argument(
        "--view-table",
        action="store_true",
        help="Display the results table in formatted ASCII and exit",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Limit number of rows shown with --view-table",
    )

    args = parser.parse_args()
    table_csv = Path(args.table) if args.table else TABLE_FILE_PATH
    project_root = Path(__file__).resolve().parent.parent

    if args.view_table:
        print_results_table(csv_path=table_csv, last_n=args.last)
        return

    # Resolve which path (if any) the user is pointing at -- CLI --path wins,
    # then the module-level INSTANCE_PATH, in that order.
    effective_path = args.path if args.path is not None else INSTANCE_PATH

    if effective_path is not None:
        resolved_path = Path(effective_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path

        if resolved_path.is_dir():
            # A directory was given -- run every instance file inside it.
            run_directory(
                instance_dir=resolved_path,
                save_to_table=not args.no_save,
                table_path=table_csv,
            )
            return

        # A single file was given via --path -- run just that one.
        instance_path, instance_name, instance, metadata = load_single_file(resolved_path)
        compare_instance(
            instance_path=instance_path,
            instance_name=instance_name,
            instance=instance,
            metadata=metadata,
            save_to_table=not args.no_save,
            table_path=table_csv,
        )
        return

    # No path given at all -- run exactly ONE instance, by name (positional
    # arg if given, otherwise the module-level INSTANCE_NAME default).
    instance_path, instance_name, instance, metadata = load_test_instance(
        custom_name=args.instance_name,
    )
    compare_instance(
        instance_path=instance_path,
        instance_name=instance_name,
        instance=instance,
        metadata=metadata,
        save_to_table=not args.no_save,
        table_path=table_csv,
    )


if __name__ == "__main__":
    main()