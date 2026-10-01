"""
solve.py -- WOA + LNS + CP-SAT hybrid

Pipeline:
  1. Run CP-SAT on the whole instance.
  2. If OPTIMAL: done, nothing can beat it.
  3. If FEASIBLE: use its incumbent as a WOA seed (one population
     slot, not the whole population -- see woa.py's seed_chromosome
     docstring for why). Run WOA+LNS, then keep whichever of
     (CP-SAT's answer, WOA+LNS's answer) has the better makespan.
  4. If UNKNOWN/INFEASIBLE: no seed available. Run WOA+LNS from its
     normal (heuristic + random) population init, and return its
     result directly, since there's nothing from CP-SAT to compare
     against or fall back to.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .encoding import (
    decode_chromosome,
    encode_schedule,
    build_schedule_from_orders,
    get_operation_list,
)
from .woa import run_woa
from .lns import lns_polish


def _find_critical_path_standalone(schedule_by_op, machine_order, instance):
    """
    Critical path trace , sent to LNS to tweak the schedule for probable better results
    """
    jobs = instance["jobs"]

    machine_predecessor = {}
    for machine_id, ordered_ops in machine_order.items():
        for idx in range(1, len(ordered_ops)):
            machine_predecessor[ordered_ops[idx]] = ordered_ops[idx - 1]

    job_finish_time = {
        job_id: schedule_by_op[(job_id, operation_ids[-1])]["finish"]
        for job_id, operation_ids in jobs.items()
    }
    makespan = max(job_finish_time.values())
    critical_job_id = next(
        job_id for job_id, finish in job_finish_time.items() if finish == makespan
    )

    path = []
    current_operation = (critical_job_id, jobs[critical_job_id][-1])

    while current_operation is not None:
        path.append(current_operation)
        job_id, op_id = current_operation
        start_time = schedule_by_op[current_operation]["start"]

        operation_ids = jobs[job_id]
        idx_in_job = operation_ids.index(op_id)
        job_predecessor = None
        if idx_in_job > 0:
            candidate = (job_id, operation_ids[idx_in_job - 1])
            if schedule_by_op[candidate]["finish"] == start_time:
                job_predecessor = candidate

        machine_pred_candidate = machine_predecessor.get(current_operation)
        machine_predecessor_op = None
        if machine_pred_candidate is not None:
            if schedule_by_op[machine_pred_candidate]["finish"] == start_time:
                machine_predecessor_op = machine_pred_candidate

        current_operation = machine_predecessor_op if machine_predecessor_op is not None else job_predecessor

    path.reverse()
    return path


def lns_only_polish(chromosome, instance, deadline=None, lns_window_size=10, lns_time_limit=2.0):
    """
    The local_search_fn handed to run_woa(). Decodes a chromosome,
    runs ONE LNS step on its critical path, re-encodes the result.
    This replaces tabu_polish() entirely -- there is no Tabu Search
    loop here, just a single LNS call per invocation.
    """
    decoded = decode_chromosome(chromosome, instance)
    result = build_schedule_from_orders(
        decoded["chosen_machine"], decoded["machine_order"], instance
    )
    critical_path = _find_critical_path_standalone(
        result["schedule_by_op"], decoded["machine_order"], instance
    )
    polished_cm, polished_mo, polished_makespan = lns_polish(
        decoded["chosen_machine"],
        decoded["machine_order"],
        critical_path,
        instance,
        window_size=lns_window_size,
        time_limit=lns_time_limit,
        deadline=deadline,
    )

    polished_result = build_schedule_from_orders(polished_cm, polished_mo, instance)
    polished_chromosome = encode_schedule(
        polished_cm, polished_result["process_order"], instance
    )
    return polished_chromosome, polished_makespan


def solve_fjsp(
    instance: Dict[str, Any],
    cpsat_time_budget: float = 15.0,
    hybrid_time_budget: float = 15.0,
    woa_population_size: int = 30,
    woa_max_iterations: int = 100,
    lns_frequency: int = 5,
    lns_window_size: int = 10,
    lns_time_limit: float = 2.0,
    seed: int = 0,
    initial_cpsat_result: Optional[Dict[str, Any]] = None,
    skip_cpsat: bool = False,
) -> Dict[str, Any]:
    """
    Top-level entry point. Returns a dict with the final schedule plus
    a "method" tag: "cpsat_optimal", "cpsat_feasible_only",
    "hybrid_improved", or "hybrid_unseeded" (CP-SAT gave nothing to
    seed with -- UNKNOWN/INFEASIBLE, OR it raised because it found no
    feasible schedule at all within its time budget).
    """
    from .cp_sat import solve_cp_sat

    cpsat_result = initial_cpsat_result
    cpsat_dicts = None
    have_cpsat_answer = False

    if not skip_cpsat and cpsat_result is None:
        try:
            cpsat_result = solve_cp_sat(instance, max_time_in_seconds=cpsat_time_budget)
        except RuntimeError as error:
            if "CP-SAT did not find a feasible schedule" not in str(error):
                raise

    if cpsat_result is not None and cpsat_result.get("status") in {"OPTIMAL", "FEASIBLE"}:
        cpsat_dicts = _cpsat_result_to_schedule_dicts(cpsat_result, instance)
        if cpsat_result["status"] == "OPTIMAL":
            return {**cpsat_dicts, "method": "cpsat_optimal"}

        have_cpsat_answer = cpsat_result["status"] == "FEASIBLE"

    seed_chromosome = None
    if have_cpsat_answer:
        seed_chromosome = encode_schedule(
            cpsat_dicts["chosen_machine"], cpsat_dicts["process_order"], instance
        )

    best_chromosome, best_makespan, _ = run_woa(
        instance,
        population_size=woa_population_size,
        max_iterations=woa_max_iterations,
        local_search_fn=lambda c, i, **kw: lns_only_polish(
            c,
            i,
            lns_window_size=lns_window_size,
            lns_time_limit=lns_time_limit,
            **kw,
        ),
        local_search_frequency=lns_frequency,
        max_time_seconds=hybrid_time_budget,
        seed=seed,
        seed_chromosome=seed_chromosome,
    )
    hybrid_dicts = decode_chromosome(best_chromosome, instance)

    if have_cpsat_answer:
        if hybrid_dicts["makespan"] < cpsat_dicts["makespan"]:
            return {**hybrid_dicts, "method": "hybrid_improved"}
        return {**cpsat_dicts, "method": "cpsat_feasible_only"}

    return {**hybrid_dicts, "method": "hybrid_unseeded"}

def _cpsat_result_to_schedule_dicts(cpsat_result: Dict[str, Any], instance: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter: your solve_cp_sat()'s flat tuple format -> the dict shape used everywhere else."""
    schedule_by_op = {}
    chosen_machine = {}

    for job_id, op_id, machine_id, start, finish in cpsat_result.get("schedule", []):
        op = (job_id, op_id)
        schedule_by_op[op] = {"start": start, "finish": finish, "machine": machine_id}
        chosen_machine[op] = machine_id

    machine_order = {m: [] for m in instance["machines"]}
    for op, info in sorted(schedule_by_op.items(), key=lambda kv: kv[1]["start"]):
        machine_order[info["machine"]].append(op)

    process_order = [op for op, _ in sorted(schedule_by_op.items(), key=lambda kv: kv[1]["start"])]

    return {
        "schedule_by_op": schedule_by_op,
        "chosen_machine": chosen_machine,
        "machine_order": machine_order,
        "process_order": process_order,
        "makespan": cpsat_result.get("makespan"),
    }