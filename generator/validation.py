"""
Well-formedness checks for a generated FJSP instance
"""


class InstanceValidationError(Exception):
    """Raised when a generated instance violates a well-formedness rule."""


def assert_well_formed(instance: dict) -> None:
    jobs = instance["jobs"]
    machines = instance["machines"]
    E = instance["E"]
    P = instance["P"]

    machine_set = set(machines)

    # --- job/operation IDs contiguous, no duplicates --------------------
    expected_job_ids = list(range(1, len(jobs) + 1))
    if sorted(jobs.keys()) != expected_job_ids:
        raise InstanceValidationError(
            f"job IDs not contiguous starting at 1: {sorted(jobs.keys())}"
        )

    for j, ops in jobs.items():
        expected_op_ids = list(range(1, len(ops) + 1))
        if ops != expected_op_ids:
            raise InstanceValidationError(
                f"job {j} operation IDs not a contiguous chain 1..k: {ops}"
            )

    # --- every operation has >=1 eligible machine ------------------------
    for j, ops in jobs.items():
        for k in ops:
            if (j, k) not in E:
                raise InstanceValidationError(f"O({j},{k}) missing from E")
            eligible = E[(j, k)]
            if len(eligible) < 1:
                raise InstanceValidationError(
                    f"O({j},{k}) has no eligible machine"
                )
            if len(eligible) != len(set(eligible)):
                raise InstanceValidationError(
                    f"O({j},{k}) has duplicate machines in E: {eligible}"
                )
            if not set(eligible).issubset(machine_set):
                raise InstanceValidationError(
                    f"O({j},{k}) eligible set references unknown machine(s): "
                    f"{set(eligible) - machine_set}"
                )

    # --- processing time defined for every eligible machine, and ONLY
    #     for eligible machines; strictly positive integers ------------
    for j, ops in jobs.items():
        for k in ops:
            eligible = set(E[(j, k)])
            for mach in eligible:
                if (j, k, mach) not in P:
                    raise InstanceValidationError(
                        f"missing processing time p({j},{k},{mach})"
                    )
                p_val = P[(j, k, mach)]
                if not isinstance(p_val, int) or p_val <= 0:
                    raise InstanceValidationError(
                        f"p({j},{k},{mach}) = {p_val!r} is not a strictly "
                        f"positive integer"
                    )
            for mach in machine_set - eligible:
                if (j, k, mach) in P:
                    raise InstanceValidationError(
                        f"p({j},{k},{mach}) defined for an ineligible machine"
                    )

    # --- total operations matches sum of k_j ------------------------------
    total_ops_expected = sum(len(ops) for ops in jobs.values())
    total_ops_in_E = len(E)
    if total_ops_expected != total_ops_in_E:
        raise InstanceValidationError(
            f"operation count mismatch: jobs imply {total_ops_expected}, "
            f"E has {total_ops_in_E}"
        )
