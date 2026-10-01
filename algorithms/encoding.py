"""
encoding.py -- the bridge between three different ways of representing
a candidate solution to the FJSP:

  1. CHROMOSOME  - what WOA actually moves around. A flat list of 2N
                    real numbers in [0, 1], where N = total number of
                    operations in the instance. The first N numbers are
                    "machine selection" (MS) genes, the last N are
                    "operation sequence" (OS) genes. This is the ONLY
                    representation WOA's search equations touch.

  2. ORDER STATE -- what LNS actually edits. Two plain dictionaries:
                    which machine each operation is assigned to
                    (chosen_machine), and in what order each machine
                    processes its operations (machine_order). This is
                    a much more "editable" representation than a flat
                    vector -- LNS wants to freeze most of it and let
                    CP-SAT re-decide a small window directly.

  3. TIMED SCHEDULE -- the actual answer: for every operation, when it
                    starts and finishes, and on which machine. This is
                    what gets handed to the validator, and what the
                    makespan (our objective) is computed from.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple


#O(3,5) 5th operation of job 3
OperationId = Tuple[int, int]

# 1. A fixed, canonical ordering of every operation in the instance.
def get_operation_list(instance: Dict[str, Any]) -> List[OperationId]:
    """
    Returns every operation in the instance in one fixed order:
    job 1's operations first, then job 2's, and so on.

    This fixed order gives each gene in the chromosome a stable meaning:
    gene index 7 always corresponds to the 8th operation in this list.
    """
    jobs = instance["jobs"]
    operation_list: List[OperationId] = []
    for job_id in sorted(jobs.keys()):
        for operation_id in jobs[job_id]:
            operation_list.append((job_id, operation_id))
    return operation_list
#[(1,1),(1,2),(1,3),(2,1),(2,2)....]

# Helper for inserting an operation into a machine's timeline at the
# earliest feasible gap, instead of always appending.
def _find_active_insertion_point(
    busy_windows: List[Tuple[int, int]],
    earliest_start: int,
    duration: int,
) -> int:
    """
    Given a machine's current list of (start, finish) busy windows
    (sorted by start time), finds the earliest time >= earliest_start
    where an operation of the given duration can be inserted without
    overlapping anything already scheduled.

    Walks through each existing window in order, checking whether
    there's enough room in the gap immediately before it. If none of
    the gaps between (or before) existing windows fit, falls back to
    after everything.
    """
    cursor = earliest_start
    for (busy_start, busy_finish) in busy_windows:
        if cursor + duration <= busy_start:
            # There's room in the gap right here, before this window.
            return cursor
        # No room before this window, jump past it and keep looking.
        cursor = max(cursor, busy_finish)
    # No earlier gap fit hence goes after everything.
    return cursor

# 2. CHROMOSOME -> ORDER STATE + TIMED SCHEDULE (decode step)

def decode_chromosome(chromosome: List[float], instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turns one whale's chromosome into an actual, guaranteed-valid,
    fully timed schedule. This is the single most important function
    in the whole algorithm -- WOA's "fitness" is ->
    "decode this chromosome, then look at the resulting makespan".

    How it works->

      Step A - work out which machine each operation uses.
        The first half of the chromosome (the "MS" genes) are numbers
        in [0,1], one per operation. For an operation with e.g. 3
        eligible machines, we chop [0,1] into 3 equal buckets and see
        which bucket the gene falls into, that bucket's machine is
        the one this operation will use. Every gene value, no matter
        what it is, always lands in SOME valid bucket so this step
        can never produce an illegal machine assignment.

      Step B - decide the order operations get scheduled in.
        The second half of the chromosome (the "OS" genes) are also
        numbers in [0,1], one per operation". Lower gene value number means
        "try to schedule me sooner". But a job's operations must run
        in order, so at any point only ONE operation per job is
        actually eligible to be scheduled next. We repeatedly look at
        all currently-eligible operations (one candidate per job that
        still has operations left) and schedule whichever one has the
        smallest priority gene value. This can never violate job
        precedence, because an operation only ever becomes a candidate
        once its predecessor has already been scheduled.

      Step C - work out the actual start/finish times (ACTIVE
        scheduling). For each operation, we know the earliest it could
        possibly start (its job's ready time). Rather than always
        placing it after everything already scheduled on its chosen
        machine, we scan that machine's timeline for the earliest gap
        -- on or after the job-ready time -- that's big enough to fit
        it, and use that. Only if no such gap exists do we place it
        after everything else. This still guarantees no two operations
        share a machine at the same time, and a job's operations never
        run out of order -- it just also avoids leaving usable idle
        time on the table.

    Returns a dictionary with:
      - "schedule_by_op": {(job_id, op_id): {"start", "finish", "machine"}}
      - "chosen_machine": {(job_id, op_id): machine_id}
      - "machine_order":  {machine_id: [operations in CHRONOLOGICAL
                           (start-time) order]}
      - "process_order":  every operation, in the exact global order it
                           was DECIDED (used later for re-encoding).
      - "makespan": the resulting C_max
    """
    operation_list = get_operation_list(instance)
    n_operations = len(operation_list)
    #First n
    ms_genes = chromosome[:n_operations]
    #Last n
    os_genes = chromosome[n_operations:]

    eligible_machines = instance["E"]
    processing_times = instance["P"]
    jobs = instance["jobs"]

    #  Step A: decode the machine-selection genes into a concrete
    #     machine choice for every operation.
    chosen_machine: Dict[OperationId, int] = {}
    priority: Dict[OperationId, float] = {}
    for position, operation in enumerate(operation_list):
        eligible = eligible_machines[operation]
        gene_value = ms_genes[position]

        bucket_index = int(gene_value * len(eligible))
        if bucket_index >= len(eligible):
            # gene_value == 1.0 exactly would otherwise land one bucket
            # past the end -- clip it back onto the last valid machine.
            bucket_index = len(eligible) - 1

        chosen_machine[operation] = eligible[bucket_index]
        priority[operation] = os_genes[position]

    # Steps B + C: greedily build the schedule, always picking the
    #     eligible operation with the smallest priority ticket next,
    #     and placing it into the EARLIEST feasible gap on its machine. -
    job_next_index = {job_id: 0 for job_id in jobs}       # which op is next per job
    job_ready_time = {job_id: 0 for job_id in jobs}        # when that job is free to continue

    # Each machine's current list of (start, finish) busy windows,
    # kept sorted by start time , this is what lets us find gaps
    # instead of only ever knowing "the last thing ended at X".
    machine_busy_windows: Dict[int, List[Tuple[int, int]]] = {
        m: [] for m in instance["machines"]
    }

    schedule_by_op: Dict[OperationId, Dict[str, int]] = {}
    process_order: List[OperationId] = []

    remaining_operations = n_operations
    while remaining_operations > 0:
        # Every job that still has operations left contributes exactly
        # one candidate: whichever operation comes next in that job's
        # chain. This is the set of operations legally allowed to run
        # right now, without breaking job precedence.
        candidates = []
        for job_id, operation_ids in jobs.items():
            next_index = job_next_index[job_id]
            if next_index < len(operation_ids):
                candidates.append((job_id, operation_ids[next_index]))

        # Among those candidates, schedule whichever has the smallest
        # priority ticket -- this is the "random key" decoding rule.
        chosen_operation = min(candidates, key=lambda op: priority[op])

        job_id, op_id = chosen_operation
        machine = chosen_machine[chosen_operation]
        duration = processing_times[(job_id, op_id, machine)]

        # find the earliest feasible gap on this
        # machine, on or after the job's ready time, instead of always
        # appending after everything already on the machine.
        earliest_start = job_ready_time[job_id]
        start_time = _find_active_insertion_point(
            machine_busy_windows[machine], earliest_start, duration
        )
        finish_time = start_time + duration

        schedule_by_op[chosen_operation] = {
            "start": start_time,
            "finish": finish_time,
            "machine": machine,
        }

        # Insert into the sorted busy-windows list at the correct
        # position (not necessarily the end, since we may have just
        # filled an earlier gap rather than appended).
        windows = machine_busy_windows[machine]
        insert_at = 0
        while insert_at < len(windows) and windows[insert_at][0] < start_time:
            insert_at += 1
        windows.insert(insert_at, (start_time, finish_time))

        process_order.append(chosen_operation)

        job_ready_time[job_id] = finish_time
        job_next_index[job_id] += 1
        remaining_operations -= 1

    # machine_order must be built from actual chronological (start-time)
    # order, since operations are no longer necessarily scheduled onto
    # a machine in the same order they were decided -- an operation
    # decided later can still land in an earlier time slot than one
    # decided earlier.
    machine_order: Dict[int, List[OperationId]] = {m: [] for m in instance["machines"]}
    for op, info in schedule_by_op.items():
        machine_order[info["machine"]].append(op)
    for m in machine_order:
        machine_order[m].sort(key=lambda op: schedule_by_op[op]["start"])

    makespan = max(entry["finish"] for entry in schedule_by_op.values())

    return {
        "schedule_by_op": schedule_by_op,
        "chosen_machine": chosen_machine,
        "machine_order": machine_order,
        "process_order": process_order,
        "makespan": makespan,
    }


# 3. ORDER STATE -> TIMED SCHEDULE (used by LNS after it puts a
#    solved window back into a full schedule)
def validate_order_state(
    chosen_machine: Dict[OperationId, int],
    machine_order: Dict[int, List[OperationId]],
    instance: Dict[str, Any],
) -> None:
    """
    Validate that the explicit order state is internally consistent before
    trying to rebuild a timed schedule from it.

    This is a defensive check: it catches malformed search states early
    and gives a clear error instead of letting a later step fail in a less
    helpful way.
    """
    expected_operations = set(get_operation_list(instance))
    eligible_machines = instance["E"]
    machine_ids = set(instance["machines"])

    if set(chosen_machine.keys()) != expected_operations:
        missing = sorted(expected_operations - set(chosen_machine.keys()))
        extra = sorted(set(chosen_machine.keys()) - expected_operations)
        details = []
        if missing:
            details.append(f"missing operations: {missing}")
        if extra:
            details.append(f"unexpected operations: {extra}")
        raise ValueError(
            "chosen_machine is inconsistent with the instance. " + "; ".join(details)
        )

    for operation, machine_id in chosen_machine.items():
        if machine_id not in eligible_machines[operation]:
            raise ValueError(
                f"chosen_machine assigns O({operation[0]},{operation[1]}) to "
                f"machine {machine_id}, but that machine is not eligible for the operation."
            )

    if set(machine_order.keys()) != machine_ids:
        missing = sorted(machine_ids - set(machine_order.keys()))
        extra = sorted(set(machine_order.keys()) - machine_ids)
        details = []
        if missing:
            details.append(f"missing machines: {missing}")
        if extra:
            details.append(f"unexpected machines: {extra}")
        raise ValueError(
            "machine_order is missing or has extra machine keys. " + "; ".join(details)
        )

    seen_operations = set()
    for machine_id, ordered_ops in machine_order.items():
        if not isinstance(ordered_ops, list):
            raise ValueError(
                f"machine_order[{machine_id}] must be a list of operations, not {type(ordered_ops).__name__}."
            )

        for operation in ordered_ops:
            if operation not in expected_operations:
                raise ValueError(
                    f"machine_order[{machine_id}] contains an unknown operation {operation}."
                )
            if chosen_machine.get(operation) != machine_id:
                raise ValueError(
                    f"machine_order[{machine_id}] contains O({operation[0]},{operation[1]}), "
                    f"but chosen_machine assigns that operation to machine {chosen_machine.get(operation)}."
                )
            if operation in seen_operations:
                raise ValueError(
                    f"Operation O({operation[0]},{operation[1]}) appears more than once across machine_order."
                )
            seen_operations.add(operation)

    if seen_operations != expected_operations:
        missing = sorted(expected_operations - seen_operations)
        raise ValueError(
            "machine_order does not cover every operation exactly once. "
            f"Missing operations: {missing}"
        )


def build_schedule_from_orders(
    chosen_machine: Dict[OperationId, int],
    machine_order: Dict[int, List[OperationId]],
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Re-times a schedule from an EXPLICIT machine assignment and an
    EXPLICIT per-machine processing order, instead of decoding from a
    chromosome. LNS needs this: after putting a CP-SAT-solved window
    back into a larger schedule, machine_order and chosen_machine have
    been edited directly ie here's no chromosome involved at that
    point, so decode_chromosome() doesn't apply. This function answers
    "given these two explicit ordering decisions, what are the
    resulting start/finish times?"

    Unlike decode_chromosome (which is free to pick whichever globally
    eligible operation has the best priority, and therefore can
    never create a contradiction), this function is handed ordering
    decisions from OUTSIDE, and those decisions CAN contradict each other. 
    To detect that safely, we do a proper
    TOPOLOGICAL pass: schedule an operation only once everything that
    must come before it (its job-predecessor AND its machine-
    predecessor) has already been scheduled. If we ever get stuck with
    operations left over that never become eligible, the ordering was
    contradictory, and we say so clearly rather than producing a
    silently wrong schedule.

    Returns the same shape of dictionary as decode_chromosome (minus
    the chromosome-specific "process_order" naming, though it's still
    included), or raises ValueError if the supplied orders contradict
    each other (this is expected to happen occasionally after an LNS
    splice callers should catch it and simply discard that
    candidate result).
    """
    validate_order_state(chosen_machine, machine_order, instance)

    jobs = instance["jobs"]
    processing_times = instance["P"]

    all_operations = list(chosen_machine.keys())

    # Every operation has up to two predecessors it must wait for:
    # the previous operation in its own job's chain, and the previous
    # operation in its assigned machine's processing order.
    job_predecessor: Dict[OperationId, OperationId] = {}
    for job_id, operation_ids in jobs.items():
        for idx in range(1, len(operation_ids)):
            job_predecessor[(job_id, operation_ids[idx])] = (job_id, operation_ids[idx - 1])

    machine_predecessor: Dict[OperationId, OperationId] = {}
    for machine_id, ordered_ops in machine_order.items():
        for idx in range(1, len(ordered_ops)):
            machine_predecessor[ordered_ops[idx]] = ordered_ops[idx - 1]

    # Standard "Kahn's algorithm" style topological processing: track
    # how many un-scheduled predecessors each operation still has left,
    # and only make an operation eligible once that count hits zero.
    remaining_predecessor_count: Dict[OperationId, int] = {}
    successors: Dict[OperationId, List[OperationId]] = {op: [] for op in all_operations}

    for operation in all_operations:
        predecessors = []
        if operation in job_predecessor:
            predecessors.append(job_predecessor[operation])
        if operation in machine_predecessor:
            predecessors.append(machine_predecessor[operation])

        remaining_predecessor_count[operation] = len(predecessors)
        for predecessor in predecessors:
            successors[predecessor].append(operation)

    # start with indegree zero
    ready_queue = deque(
        op for op in all_operations if remaining_predecessor_count[op] == 0
    )

    schedule_by_op: Dict[OperationId, Dict[str, int]] = {}
    process_order: List[OperationId] = []

    while ready_queue:
        operation = ready_queue.popleft()
        job_id, op_id = operation
        machine = chosen_machine[operation]
        duration = processing_times[(job_id, op_id, machine)]

        earliest_start = 0
        if operation in job_predecessor:
            earliest_start = max(earliest_start, schedule_by_op[job_predecessor[operation]]["finish"])
        if operation in machine_predecessor:
            earliest_start = max(earliest_start, schedule_by_op[machine_predecessor[operation]]["finish"])

        finish_time = earliest_start + duration
        schedule_by_op[operation] = {
            "start": earliest_start,
            "finish": finish_time,
            "machine": machine,
        }
        process_order.append(operation)

        for successor in successors[operation]:
            remaining_predecessor_count[successor] -= 1
            if remaining_predecessor_count[successor] == 0:
                ready_queue.append(successor)

    if len(process_order) != len(all_operations):
        # Some operations never became eligible -- the supplied orders
        # contradict each other (a cycle in the combined job+machine
        # precedence graph). This is a legitimate outcome of a bad
        # LNS splice,the caller should catch this and
        # simply discard that candidate result.
        raise ValueError(
            "contradictory ordering: job precedence and machine order "
            "cannot both be satisfied (a cycle was formed)"
        )

    makespan = max(entry["finish"] for entry in schedule_by_op.values())

    return {
        "schedule_by_op": schedule_by_op,
        "chosen_machine": chosen_machine,
        "machine_order": machine_order,
        "process_order": process_order,
        "makespan": makespan,
    }

# 4. ORDER STATE -> CHROMOSOME (re-encoding, so LNS's improvements can
#    be fed back into WOA's population)

def encode_schedule(
    chosen_machine: Dict[OperationId, int],
    process_order: List[OperationId],
    instance: Dict[str, Any],
) -> List[float]:
    """
    The reverse of decode_chromosome: given a concrete machine
    assignment, produce a chromosome. 
    This is how a schedule LNS has polished (or a CP-SAT
    incumbent used as a seed) gets handed back to WOA as WOA only ever
    works with chromosomes, so any schedule found outside of it needs
    to be translated back into this format before WOA can use it.

    The machine-selection genes: for an operation assigned to
    the machine at index i out of its eligible list, we just need a
    gene value that lands in bucket i we use the middle of that
    bucket, e.g. bucket i out of 3 gets gene value (i + 0.5) / 3. This
    is a completely arbitrary but safe choice within the bucket -- any
    value in that range decodes to the same machine.
    """
    operation_list = get_operation_list(instance)
    eligible_machines = instance["E"]
    n_operations = len(operation_list)

    operation_to_gene_index = {op: idx for idx, op in enumerate(operation_list)}

    ms_genes = [0.0] * n_operations
    os_genes = [0.0] * n_operations

    for operation, machine in chosen_machine.items():
        gene_index = operation_to_gene_index[operation]
        eligible = eligible_machines[operation]
        bucket_index = eligible.index(machine)
        # midpoint of the bucket -- safely decodes back to this machine
        ms_genes[gene_index] = (bucket_index + 0.5) / len(eligible)

    for position, operation in enumerate(process_order):
        gene_index = operation_to_gene_index[operation]
        # evenly spaced increasing values, one per position in the
        # global processing order
        os_genes[gene_index] = (position + 0.5) / n_operations

    return ms_genes + os_genes

# 5. TIMED SCHEDULE -> the flat list format validator_fn.py expects
def schedule_to_entries(schedule_by_op: Dict[OperationId, Dict[str, int]]) -> List[tuple]:
    """
    Converts our internal schedule_by_op dictionary into the flat
    (job_id, operation_id, machine_id, start_time, completion_time)
    tuple format that algorithms.validator_fn.validate_schedule()
    expects. Purely a formatting convenience so the algorithms and the
    Part B validator can talk to each other directly.
    """
    entries = []
    for (job_id, operation_id), timing in schedule_by_op.items():
        entries.append((
            job_id,
            operation_id,
            timing["machine"],
            timing["start"],
            timing["finish"],
        ))
    return entries