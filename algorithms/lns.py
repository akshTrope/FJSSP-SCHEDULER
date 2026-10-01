"""
lns.py -- Large Neighborhood Search. Freezes most of the schedule,
lets CP-SAT fix a small window near the critical path, splices the
result back in.
"""

import time

from ortools.sat.python import cp_model
from .encoding import build_schedule_from_orders

#Critical path-> the path which delivers makespan of the schedule we want to improve by LNS
def pick_window(critical_path, chosen_machine, machine_order, instance, window_size=10):
    """
    Chooses which operations to "unfreeze": the critical path itself,
    plus each critical op's immediate neighbors (the op right before
    and after it, both in its job and on its machine). Neighbors are
    included so CP-SAT has room to actually re-order things, not just
    re-time the same fixed order.
    """
    jobs = instance["jobs"]
    window = set()

    def add(op):
        if len(window) < window_size:
            window.add(op)

    for op in critical_path:
        add(op)

    for op in list(critical_path):
        job_id, op_id = op
        op_ids = jobs[job_id]
        idx = op_ids.index(op_id)
        if idx > 0:
            add((job_id, op_ids[idx - 1]))
        if idx < len(op_ids) - 1:
            add((job_id, op_ids[idx + 1]))

        m = chosen_machine[op]
        #which machine chose this operation
        queue = machine_order[m]
        pos = queue.index(op)
        if pos > 0:
            add(queue[pos - 1])
        if pos < len(queue) - 1:
            add(queue[pos + 1])

    return window
#Why have we added neighbours? 
# By including the neighbors,CP-SAT gets more freedom: it can, for example, decide to run the machine-neighbor after the critical operation instead of before it, 
#potentially freeing up an earlier gap for the critical operation to slide into.

def lns_polish(
    chosen_machine,
    machine_order,
    critical_path,
    instance,
    window_size=10,
    time_limit=2.0,
    deadline=None,
):
    """
    One LNS step. Returns a possibly improved (chosen_machine,
    machine_order, makespan). If CP-SAT can't improve things in time,
    just returns what was passed in, unchanged.
    """
    if deadline is not None:
        time_limit = min(time_limit, max(0.0, deadline - time.perf_counter()))

    current = build_schedule_from_orders(chosen_machine, machine_order, instance)
    schedule_by_op = current["schedule_by_op"]

    free_ops = pick_window(critical_path, chosen_machine, machine_order, instance, window_size)

    eligible_machines = instance["E"]
    processing_time = instance["P"]
    jobs = instance["jobs"]

    model = cp_model.CpModel()
    horizon = max(v["finish"] for v in schedule_by_op.values()) + 1

    start, end, machine_var = {}, {}, {}
    machine_intervals = {m: [] for m in instance["machines"]}

    # Free operations: real variables, same as in full_cpsat.py.
    for op in free_ops:
        job_id, op_id = op
        start[op] = model.NewIntVar(0, horizon, f"s_{job_id}_{op_id}")
        end[op] = model.NewIntVar(0, horizon, f"e_{job_id}_{op_id}")
        eligible = eligible_machines[op]
        machine_var[op] = model.NewIntVar(0, len(eligible) - 1, f"m_{job_id}_{op_id}")

        for idx, m in enumerate(eligible):
            is_this_one = model.NewBoolVar(f"on_{m}_{job_id}_{op_id}")
            model.Add(machine_var[op] == idx).OnlyEnforceIf(is_this_one)
            model.Add(machine_var[op] != idx).OnlyEnforceIf(is_this_one.Not())
            duration = processing_time[(job_id, op_id, m)]
            iv = model.NewOptionalIntervalVar(start[op], duration, end[op], is_this_one, f"iv_{m}_{job_id}_{op_id}")
            machine_intervals[m].append(iv)

    # Frozen operations that share a machine with a free op: fixed
    # "obstacles",not variables, just blocked time windows.
    frozen_neighbors = set()
    for op in free_ops:
        for m in instance["E"][op]:
            for other in machine_order[m]:
                if other not in free_ops:
                    frozen_neighbors.add(other)

    for op in frozen_neighbors:
        info = schedule_by_op[op]
        m = chosen_machine[op]
        fixed_iv = model.NewIntervalVar(info["start"], info["finish"] - info["start"], info["finish"], f"frozen_{op}")
        machine_intervals[m].append(fixed_iv)

    for m, intervals in machine_intervals.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Job order: free ops must respect their job neighbors, whether
    # that neighbor is free (a variable) or frozen (a fixed number).
    for op in free_ops:
        job_id, op_id = op
        op_ids = jobs[job_id]
        idx = op_ids.index(op_id)
        if idx > 0:
            pred = (job_id, op_ids[idx - 1])
            if pred in free_ops:
                model.Add(start[op] >= end[pred])
            else:
                model.Add(start[op] >= schedule_by_op[pred]["finish"])
        if idx < len(op_ids) - 1:
            succ = (job_id, op_ids[idx + 1])
            if succ in free_ops:
                model.Add(start[succ] >= end[op])
            else:
                model.Add(end[op] <= schedule_by_op[succ]["start"])

    window_makespan = model.NewIntVar(0, horizon, "window_makespan")
    model.AddMaxEquality(window_makespan, list(end.values()))
    model.Minimize(window_makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # couldn't improve this window in time -- just return what we had
        return chosen_machine, machine_order, current["makespan"]

    # splice the solved window back into the full schedule
    new_chosen_machine = dict(chosen_machine)
    touched_machines = set()
    for op in free_ops:
        eligible = eligible_machines[op]
        new_machine = eligible[solver.Value(machine_var[op])]
        touched_machines.add(chosen_machine[op])
        touched_machines.add(new_machine)
        new_chosen_machine[op] = new_machine

    new_machine_order = {m: list(ops) for m, ops in machine_order.items()}
    for m in touched_machines:
        stay = [op for op in new_machine_order[m] if op not in free_ops]
        moved_in = [op for op in free_ops if new_chosen_machine[op] == m]
        combined = stay + moved_in
        combined.sort(key=lambda op: solver.Value(start[op]) if op in start else schedule_by_op[op]["start"])
        new_machine_order[m] = combined

    new_result = build_schedule_from_orders(new_chosen_machine, new_machine_order, instance)

    if new_result["makespan"] <= current["makespan"]:
        return new_chosen_machine, new_machine_order, new_result["makespan"]
    return chosen_machine, machine_order, current["makespan"]