"""
adapter.py -- converts between solve_cp_sat()'s flat tuple format
and the dictionary format
"""

def cpsat_result_to_schedule_dicts(cpsat_result, instance):
    schedule_by_op = {}
    chosen_machine = {}

    for job_id, op_id, machine_id, start, finish in cpsat_result["schedule"]:
        op = (job_id, op_id)
        schedule_by_op[op] = {"start": start, "finish": finish, "machine": machine_id}
        chosen_machine[op] = machine_id

    # Rebuild machine_order by sorting each machine's operations by start time.
    machine_order = {m: [] for m in instance["machines"]}
    for op, info in sorted(schedule_by_op.items(), key=lambda kv: kv[1]["start"]):
        machine_order[info["machine"]].append(op)

    process_order = [op for op, _ in sorted(schedule_by_op.items(), key=lambda kv: kv[1]["start"])]

    return {
        "schedule_by_op": schedule_by_op,
        "chosen_machine": chosen_machine,
        "machine_order": machine_order,
        "process_order": process_order,
        "makespan": cpsat_result["makespan"],
    }   