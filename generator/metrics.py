import math
from typing import Dict, List, Optional, Tuple

def _normalize_bottleneck_machines(
    bottleneck_machines: Optional[List]
) -> set[int]:
    """Normalize a list of bottleneck entries into machine ids."""
    if not bottleneck_machines:
        return set()

    normalized = set()
    for entry in bottleneck_machines:
        if isinstance(entry, tuple):
            normalized.add(entry[0])
        else:
            normalized.add(entry)
    return normalized


def compute_beta(instance: dict) -> float:
    """
    beta = average |E(j,k)| / m

    Indicates how flexible the instance is: beta -> 1 means almost every
    operation can run on almost every machine (fully flexible); beta close
    to 0 means each operation is pinned to very few machines.
    """
    E = instance["E"]
    m = len(instance["machines"])
    n_ops = len(E)

    if n_ops == 0:
        return 0.0

    total_eligible = sum(len(eligible) for eligible in E.values())
    avg_eligible = total_eligible / n_ops
    return avg_eligible / m


def compute_dv(instance: dict) -> float:
    """
    dv = (# unique processing-time values) / (total machine-assignment
          options across all operations)

    Low dv -> processing times are very similar across eligible machines
    (or there are many assignment options relative to how many distinct
    durations appear) -> many distinct schedules can share the same makespan.
    High dv -> durations vary a lot across machine choices.
    """
    P = instance["P"]
    E = instance["E"]

    total_assignment_options = sum(len(eligible) for eligible in E.values())
    if total_assignment_options == 0:
        return 0.0

    unique_durations = set(P.values())
    return len(unique_durations) / total_assignment_options


def compute_beta_dv(instance: dict) -> Tuple[float, float]:
    return compute_beta(instance), compute_dv(instance)


def compute_machine_eligibility_entropy(instance: dict) -> float:
    """
    Shannon entropy over the distribution of machine eligibility counts,
    normalized by log(m) so the result lies in [0, 1].

    Higher values mean eligibility is spread more evenly across machines;
    lower values mean eligibility is concentrated on a few machines.
    """
    machines = instance["machines"]
    E = instance["E"]
    m = len(machines)

    if m <= 1 or not E:
        return 0.0

    counts = {machine: 0 for machine in machines}
    for eligible in E.values():
        for machine in eligible:
            counts[machine] += 1

    total_count = sum(counts.values())
    if total_count == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total_count
        if p > 0:
            entropy -= p * math.log(p)

    return entropy / math.log(m)


def compute_bottleneck_coverage(
    instance: dict,
    bottleneck_machines: Optional[List] = None,
) -> float:
    """
    Fraction of all operations whose eligible set contains at least one
    designated bottleneck machine.
    """
    E = instance["E"]
    if not E:
        return 0.0

    bottleneck_ids = _normalize_bottleneck_machines(bottleneck_machines)
    if not bottleneck_ids:
        return 0.0

    covered = 0
    for eligible in E.values():
        if any(machine in bottleneck_ids for machine in eligible):
            covered += 1

    return covered / len(E)


def compute_bottleneck_intensity(
    instance: dict,
    bottleneck_machines: Optional[List] = None,
) -> float:
    """
    Average ratio of a bottleneck machine's processing time to the average
    duration of the other eligible machines for that operation.

    Lower ratios indicate a stronger bottleneck pull.
    """
    E = instance["E"]
    P = instance["P"]
    bottleneck_ids = _normalize_bottleneck_machines(bottleneck_machines)

    if not E or not bottleneck_ids:
        return 0.0

    ratios = []
    for (j, k), eligible in E.items():
        bottleneck_hits = [machine for machine in eligible if machine in bottleneck_ids]
        if not bottleneck_hits:
            continue

        for machine in bottleneck_hits:
            alt_times = [P[(j, k, m)] for m in eligible if m != machine]
            if not alt_times:
                continue
            avg_alt = sum(alt_times) / len(alt_times)
            if avg_alt > 0:
                ratios.append(P[(j, k, machine)] / avg_alt)

    if not ratios:
        return 0.0

    return sum(ratios) / len(ratios)


def compute_processing_time_gap_ratio(instance: dict) -> float:
    """
    Average, over all operations, of min(time) / max(time) across the
    operation's eligible machines.

    This measures how much the fastest eligible machine differs from the
    slowest one. Values close to 1 indicate very little time gap; smaller
    values indicate a strong spread in processing times.
    """
    E = instance["E"]
    P = instance["P"]
    if not E:
        return 0.0

    ratios = []
    for (j, k), eligible in E.items():
        times = [P[(j, k, m)] for m in eligible]
        if not times:
            continue
        t_min = min(times)
        t_max = max(times)
        if t_max > 0:
            ratios.append(t_min / t_max)

    if not ratios:
        return 0.0

    return sum(ratios) / len(ratios)


def infer_bottleneck_machines(
    instance: dict,
    top_fraction: float = 0.2,
) -> list:
    """Infer machines that act as bottlenecks from eligibility and time data."""
    jobs = instance["jobs"]
    E = instance["E"]
    P = instance["P"]
    machines = instance["machines"]

    eligibility_count = {m: 0 for m in machines}
    speed_ratios = {m: [] for m in machines}

    for j, ops in jobs.items():
        for k in ops:
            eligible = E.get((j, k), [])
            if len(eligible) < 2:
                continue

            for m in eligible:
                eligibility_count[m] += 1
                alt_times = [P[(j, k, other)] for other in eligible if other != m]
                avg_alt = sum(alt_times) / len(alt_times)
                if avg_alt > 0:
                    speed_ratios[m].append(P[(j, k, m)] / avg_alt)

    total_eligibility = sum(eligibility_count.values())
    if total_eligibility == 0:
        return []

    scores = {}
    for m in machines:
        eligibility_share = eligibility_count[m] / total_eligibility
        avg_speed_ratio = (
            sum(speed_ratios[m]) / len(speed_ratios[m]) if speed_ratios[m] else 1.0
        )
        speed_deviation = abs(avg_speed_ratio - 1.0)
        scores[m] = eligibility_share * speed_deviation

    ranked = sorted(machines, key=lambda m: scores[m], reverse=True)
    top_n = max(1, round(len(machines) * top_fraction))
    return ranked[:top_n]


def compute_bottleneck_metric(
    instance: dict,
    bottleneck_machines: Optional[List] = None,
    top_fraction: float = 0.2,
) -> float:
    """Bottleneck score; the raw intensity ratio can make it exceed 1."""
    if bottleneck_machines is None:
        bottleneck_machines = infer_bottleneck_machines(instance, top_fraction=top_fraction)

    if not bottleneck_machines:
        return 0.0

    coverage = compute_bottleneck_coverage(instance, bottleneck_machines)
    intensity = compute_bottleneck_intensity(instance, bottleneck_machines)
    return 0.5 * coverage + 0.5 * intensity


def compute_time_skew(instance: dict) -> float:
    """
    Average, over all operations, of std(time) / mean(time) across the
    eligible machines for that operation.

    This is the duration coefficient of variation (CV) per operation,
    capturing how uneven the machine-specific durations are.
    """
    E = instance["E"]
    P = instance["P"]
    if not E:
        return 0.0

    skews = []
    for (j, k), eligible in E.items():
        times = [P[(j, k, m)] for m in eligible]
        if not times:
            continue

        mean_t = sum(times) / len(times)
        if mean_t == 0:
            continue

        variance = sum((time - mean_t) ** 2 for time in times) / len(times)
        std_t = math.sqrt(variance)
        skews.append(std_t / mean_t)

    if not skews:
        return 0.0

    return sum(skews) / len(skews)


def summarize_instance(
    instance: dict,
    bottleneck_machines: Optional[List] = None,
) -> Dict:
    """
    Returns a small dict of characterization numbers, useful for logging
    / comparing generated instances against real benchmark family
    averages (see Table 1 of the paper referenced above).
    """
    beta, dv = compute_beta_dv(instance)
    jobs = instance["jobs"]
    n_jobs = len(jobs)
    n_machines = len(instance["machines"])
    n_ops = sum(len(ops) for ops in jobs.values())
    if bottleneck_machines is None:
        bottleneck_machines = infer_bottleneck_machines(instance)

    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "n_operations": n_ops,
        "avg_ops_per_job": n_ops / n_jobs if n_jobs else 0.0,
        "beta": beta,
        "dv": dv,
        "machine_eligibility_entropy": compute_machine_eligibility_entropy(instance),
        "bottleneck_coverage": compute_bottleneck_coverage(instance, bottleneck_machines),
        "bottleneck_intensity": compute_bottleneck_intensity(instance, bottleneck_machines),
        "bottleneck_metric": compute_bottleneck_metric(
            instance,
            bottleneck_machines=bottleneck_machines,
        ),
        "processing_time_gap_ratio": compute_processing_time_gap_ratio(instance),
        "time_skew": compute_time_skew(instance),
    }


__all__ = [
    "compute_beta",
    "compute_dv",
    "compute_beta_dv",
    "compute_machine_eligibility_entropy",
    "compute_bottleneck_coverage",
    "compute_bottleneck_intensity",
    "infer_bottleneck_machines",
    "compute_bottleneck_metric",
    "compute_processing_time_gap_ratio",
    "compute_time_skew",
    "summarize_instance",
]
