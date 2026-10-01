from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ortools.sat.python import cp_model

from generator.io_utils import load_instance


def solve_cp_sat(
    instance: Dict[str, Any],
    max_time_in_seconds: int = 30,
    num_search_workers: int = 8,
    log_search_progress: bool = False,
) -> Dict[str, Any]:
    """
    Solve the FJSP with CP-SAT and return a full schedule together with its makespan.

    The model is built around a clean and robust encoding:
      - one boolean variable per (operation, eligible_machine)
      - exactly one machine is selected for every operation
      - one start/end variable per operation
      - one optional interval per (operation, machine)
      - optional intervals are added to AddNoOverlap for each machine
      - job precedence is enforced directly
      - makespan is minimized

    Returns:
      {
        "schedule": [(job_id, op_id, machine_id, start_time, finish_time), ...],
        "makespan": int,
        "status": str,
        "time_seconds": float,
      }
    """
    jobs = instance["jobs"]
    machines = instance["machines"]
    E = instance["E"]
    P = instance["P"]

    operation_list: List[Tuple[int, int]] = []
    for job_id in sorted(jobs):
        for op_id in jobs[job_id]:
            operation_list.append((job_id, op_id))
    #create cp sat model
    model = cp_model.CpModel()
    #whether an operation is assigned to a machine or not
    x: Dict[Tuple[int, int], Dict[int, cp_model.IntVar]] = {}
    start_time: Dict[Tuple[int, int], cp_model.IntVar] = {}
    end_time: Dict[Tuple[int, int], cp_model.IntVar] = {}

    # Create one machine-selection boolean per eligible assignment.
    for operation in operation_list:
        eligible = E[operation]
        x[operation] = {}
        for machine_id in eligible:
            x[operation][machine_id] = model.NewBoolVar(
                f"x_{operation[0]}_{operation[1]}_{machine_id}"
            )
        #x[(1,2)][3]=1 operation (1,2) is assigned to machine 3 (boolean)
        #Each operation is forcefully assigned to one machine
        model.Add(sum(x[operation].values()) == 1)

        start_time[operation] = model.NewIntVar(0, 10**9, f"s_{operation[0]}_{operation[1]}")
        end_time[operation] = model.NewIntVar(0, 10**9, f"e_{operation[0]}_{operation[1]}")

        machine_duration_expr = sum(
            P[(operation[0], operation[1], machine_id)] * x[operation][machine_id]
            for machine_id in eligible
        )
        model.Add(end_time[operation] == start_time[operation] + machine_duration_expr)

    # Job precedence: operation k+1 can only start after operation k finishes.
    for job_id in sorted(jobs):
        operation_ids = jobs[job_id]
        for idx in range(len(operation_ids) - 1):
            current = (job_id, operation_ids[idx])
            next_op = (job_id, operation_ids[idx + 1])
            model.Add(start_time[next_op] >= end_time[current])

    # Machine non-overlap using optional intervals.
    machine_intervals: Dict[int, List[cp_model.IntervalVar]] = {machine_id: [] for machine_id in machines}

    for operation in operation_list:
        for machine_id in E[operation]:
            interval = model.NewOptionalIntervalVar(
                start_time[operation],
                P[(operation[0], operation[1], machine_id)],
                end_time[operation],
                x[operation][machine_id],
                f"interval_{operation[0]}_{operation[1]}_{machine_id}",
            )
            machine_intervals[machine_id].append(interval)

    for machine_id in machines:
        if machine_intervals[machine_id]:
            model.AddNoOverlap(machine_intervals[machine_id])

    # Makespan.
    makespan = model.NewIntVar(0, 10**9, "makespan")
    for operation in operation_list:
        model.Add(end_time[operation] <= makespan)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.log_search_progress = log_search_progress

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT did not find a feasible schedule. Status: {solver.StatusName(status)}"
        )

    schedule_entries: List[Tuple[int, int, int, int, int]] = []
    for operation in operation_list:
        machine_id = None
        for candidate in E[operation]:
            if solver.Value(x[operation][candidate]):
                machine_id = candidate
                break

        if machine_id is None:
            raise RuntimeError(f"No machine selected for operation O({operation[0]},{operation[1]}).")

        schedule_entries.append(
            (
                operation[0],
                operation[1],
                machine_id,
                solver.Value(start_time[operation]),
                solver.Value(end_time[operation]),
            )
        )

    return {
        "schedule": schedule_entries,
        "makespan": int(solver.Value(makespan)),
        "status": solver.StatusName(status),
        "time_seconds": solver.WallTime(),
    }


def run_cp_sat_from_file(instance_path: str, **kwargs: Any) -> Dict[str, Any]:
    """Load a generated FJSP instance JSON file and solve it with CP-SAT."""
    instance, _ = load_instance(instance_path)
    return solve_cp_sat(instance, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve an FJSP instance with CP-SAT.")
    parser.add_argument("instance", type=str, help="Path to a generated instance JSON file.")
    parser.add_argument("--time", type=int, default=30, help="Max solver time in seconds.")
    parser.add_argument("--workers", type=int, default=8, help="Search workers.")
    args = parser.parse_args()

    result = run_cp_sat_from_file(
        args.instance,
        max_time_in_seconds=args.time,
        num_search_workers=args.workers,
    )
    print(json.dumps(result, indent=2))
