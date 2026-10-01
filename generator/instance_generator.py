"""
FJSP Random Instance Generator
"""

import random
from typing import Dict, List, Optional, Tuple

from .sampling import (
    resolve_count,
    resolve_job_length,
    sample_eligible_machines,
    sample_processing_times,
    normalize_bottleneck_config,
    bottleneck_stage_matches,
    IntOrRange,
    BottleneckEntry,
)
from .metrics import summarize_instance
from .validation import assert_well_formed


def _assign_job_owners(
    distributed_bottlenecks: List[BottleneckEntry],
    default_speed_factor: Tuple[float, float],
    n_jobs: int,
) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]]]:
    # Give successive jobs a different preferred bottleneck machine in a
    # round-robin pattern. This spreads congestion instead of concentrating
    # every job on the same injected machine.
    normalized = []
    for entry in distributed_bottlenecks:
        if isinstance(entry, tuple):
            normalized.append(entry)
        else:
            normalized.append((entry, default_speed_factor))
    speed_map = {machine_id: speed_range for machine_id, speed_range in normalized}

    job_owner_machine = {}
    for j in range(1, n_jobs + 1):
        machine_id, _ = normalized[(j - 1) % len(normalized)]
        job_owner_machine[j] = machine_id

    return job_owner_machine, speed_map


def generate_instance(
    number_of_jobs: IntOrRange,
    number_of_machines: int,
    operations_per_job: IntOrRange,
    machine_flexibility: float,
    processing_time_range: Tuple[int, int],
    processing_time_variance: float,
    bottleneck_probability: float,
    seed: int,
    bottleneck_machines: Optional[List[BottleneckEntry]] = None,
    bottleneck_speed_factor: Tuple[float, float] = (0.5, 0.8),
    flexibility_mode: str = "random",
    instance_class_label: str = "custom",
    processing_time_distribution: str = "uniform",
    bottleneck_stage_filter: str = "all",
    distributed_bottlenecks: Optional[List[BottleneckEntry]] = None,
    job_length_mode: str = "uniform",
    bimodal_long_fraction: float = 0.2,
) -> Tuple[dict, dict]:
    # A local RNG makes the same seed reproduce the exact same instance.
    rng = random.Random(seed)

    machines = list(range(1, number_of_machines + 1))

    n = resolve_count(number_of_jobs, rng)
    jobs: Dict[int, List[int]] = {}
    for j in range(1, n + 1):
        # Jobs may have a fixed length or a sampled length, depending on the
        # selected job-length mode.
        k_j = resolve_job_length(
            operations_per_job, job_length_mode, rng,
            bimodal_long_fraction=bimodal_long_fraction,
        )
        jobs[j] = list(range(1, k_j + 1))

    E: Dict[Tuple[int, int], List[int]] = {}
    for j, ops in jobs.items():
        for k in ops:
            E[(j, k)] = sample_eligible_machines(
                machines=machines,
                target_beta=machine_flexibility,
                mode=flexibility_mode,
                rng=rng,
            )

    # E records where an operation is allowed to run. Processing times are
    # assigned later, once bottleneck machines have been injected into E.
    bottleneck_speed_map = normalize_bottleneck_config(
        bottleneck_machines, bottleneck_speed_factor
    )
    effective_bottleneck_machines = (
        distributed_bottlenecks
        if distributed_bottlenecks is not None
        else bottleneck_machines
        if bottleneck_machines is not None
        else ([1] if bottleneck_probability > 0 else None)
    )
    job_owner_machine: Optional[Dict[int, int]] = None
    if distributed_bottlenecks:
        job_owner_machine, distributed_speed_map = _assign_job_owners(
            distributed_bottlenecks, bottleneck_speed_factor, n
        )
        bottleneck_speed_map.update(distributed_speed_map)

    # Inject bottleneck machines only at the requested stages and with the
    # requested probability; this changes eligibility, not job precedence.
    for j, ops in jobs.items():
        k_total = len(ops)
        if job_owner_machine is not None:
            inject_machines = [job_owner_machine[j]]
        else:
            inject_machines = list(bottleneck_speed_map.keys())

        for k in ops:
            if not bottleneck_stage_matches(k, k_total, bottleneck_stage_filter):
                continue
            if rng.random() < bottleneck_probability:
                for bm in inject_machines:
                    if bm not in E[(j, k)]:
                        E[(j, k)].append(bm)

    pt_min, pt_max = processing_time_range
    P: Dict[Tuple[int, int, int], int] = {}
    for j, ops in jobs.items():
        for k in ops:
            times = sample_processing_times(
                eligible_machines=E[(j, k)],
                pt_min=pt_min,
                pt_max=pt_max,
                processing_time_variance=processing_time_variance,
                flexibility_mode=flexibility_mode,
                bottleneck_speed_map=bottleneck_speed_map,
                rng=rng,
                distribution=processing_time_distribution,
            )
            for mach, p_val in times.items():
                # P stores a duration for every eligible operation-machine pair.
                P[(j, k, mach)] = p_val

    instance = {
        "jobs": jobs,
        "machines": machines,
        "E": E,
        "P": P,
    }

    #check
    assert_well_formed(instance)

    # Store measured properties too, so experiments can be compared using the
    # instance that was actually generated rather than only its input settings.
    summary = summarize_instance(instance, effective_bottleneck_machines)

    metadata = {
        "seed": seed,
        "instance_class_label": instance_class_label,
        "params": {
            "number_of_jobs": number_of_jobs,
            "number_of_machines": number_of_machines,
            "operations_per_job": operations_per_job,
            "machine_flexibility": machine_flexibility,
            "processing_time_range": processing_time_range,
            "processing_time_variance": processing_time_variance,
            "bottleneck_probability": bottleneck_probability,
            "bottleneck_machines": effective_bottleneck_machines,
            "bottleneck_speed_factor": bottleneck_speed_factor,
            "flexibility_mode": flexibility_mode,
            "processing_time_distribution": processing_time_distribution,
            "bottleneck_stage_filter": bottleneck_stage_filter,
            "distributed_bottlenecks": distributed_bottlenecks,
            "job_length_mode": job_length_mode,
            "bimodal_long_fraction": bimodal_long_fraction,
        },
        "computed_beta": summary["beta"],
        "computed_dv": summary["dv"],
        "machine_eligibility_entropy": summary["machine_eligibility_entropy"],
        "bottleneck_coverage": summary["bottleneck_coverage"],
        "bottleneck_intensity": summary["bottleneck_intensity"],
        "processing_time_gap_ratio": summary["processing_time_gap_ratio"],
        "time_skew": summary["time_skew"],
        "n_jobs": n,
        "n_machines": number_of_machines,
        "n_operations": sum(len(ops) for ops in jobs.values()),
    }

    return instance, metadata


def regenerate(metadata: dict) -> Tuple[dict, dict]:
    # Metadata contains both the original seed and every generation setting.
    # Reusing them makes this a deterministic reconstruction, not a new sample.
    params = metadata["params"]
    return generate_instance(
        seed=metadata["seed"],
        instance_class_label=metadata["instance_class_label"],
        **params,
    )
