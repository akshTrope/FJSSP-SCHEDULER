"""
Hand-built edge cases
single machine; one job; many jobs with one bottleneck machine; extreme
time gaps; machine advantage; near-total flexibility; near-zero
flexibility; identical processing times; long critical chains; many
very short operations.

Most of these are just deliberately extreme parameter choices fed into
generate_instance().
"""

from .instance_generator import generate_instance
from .validation import assert_well_formed
from .metrics import summarize_instance

EDGE_CASE_SEED = 1  # fixed seed for all edge cases -> always reproducible


def single_machine(seed: int = EDGE_CASE_SEED):
    """m=1: every operation has exactly one possible machine."""
    return generate_instance(
        number_of_jobs=5, number_of_machines=1,
        operations_per_job=(3, 6), machine_flexibility=1.0,
        processing_time_range=(5, 50), processing_time_variance=0.2,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_single_machine",
    )


def one_job(seed: int = EDGE_CASE_SEED):
    """n=1: a single job, moderately long"""
    return generate_instance(
        number_of_jobs=1, number_of_machines=6,
        operations_per_job=(10, 10), machine_flexibility=0.4,
        processing_time_range=(5, 50), processing_time_variance=0.2,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_one_job",
    )


def single_bottleneck(seed: int = EDGE_CASE_SEED):
    """
    Many jobs, low base flexibility, bottleneck_probability=1.0 so every
    operation is forced to also be eligible on machine 1.
    """
    return generate_instance(
        number_of_jobs=20, number_of_machines=5,
        operations_per_job=(3, 6), machine_flexibility=0.1,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=1.0, seed=seed, flexibility_mode="random",
        bottleneck_machines=[1],
        instance_class_label="edge_single_bottleneck",
    )


def extreme_time_gap(seed: int = EDGE_CASE_SEED):
    """
    One operation, two eligible machines with a huge time gap
    (M1 -> 1 vs M2 -> 10000). Constructed
    directly.
    """
    instance = {
        "jobs": {1: [1]},
        "machines": [1, 2],
        "E": {(1, 1): [1, 2]},
        "P": {(1, 1, 1): 1, (1, 1, 2): 10000},
    }
    assert_well_formed(instance)
    summary = summarize_instance(instance)
    metadata = {
        "seed": seed,
        "instance_class_label": "edge_extreme_time_gap",
        "params": {"hand_built": True},
        "computed_beta": summary["beta"],
        "computed_dv": summary["dv"],
        "machine_eligibility_entropy": summary["machine_eligibility_entropy"],
        "bottleneck_coverage": summary["bottleneck_coverage"],
        "bottleneck_intensity": summary["bottleneck_intensity"],
        "processing_time_gap_ratio": summary["processing_time_gap_ratio"],
        "time_skew": summary["time_skew"],
        "n_jobs": 1,
        "n_machines": 2,
        "n_operations": 1,
    }
    return instance, metadata


def machine_advantage(seed: int = EDGE_CASE_SEED):
    """
    Generate a normal instance, then post-process: wherever machine 1
    is eligible, force its processing time down to ~30% of the smallest
    other eligible machine's time for that operation.
    """
    instance, metadata = generate_instance(
        number_of_jobs=10, number_of_machines=5,
        operations_per_job=(4, 8), machine_flexibility=0.5,
        processing_time_range=(10, 60), processing_time_variance=0.15,
        bottleneck_probability=0.5, seed=seed, flexibility_mode="random",
        bottleneck_machines=[1],
        instance_class_label="edge_machine_advantage",
    )

    for (j, k), eligible in instance["E"].items():
        if 1 in eligible:
            other_times = [
                instance["P"][(j, k, m)] for m in eligible if m != 1
            ]
            if other_times:
                instance["P"][(j, k, 1)] = max(1, round(0.3 * min(other_times)))

    assert_well_formed(instance)
    summary = summarize_instance(instance)
    metadata["computed_beta"] = summary["beta"]
    metadata["computed_dv"] = summary["dv"]
    metadata["machine_eligibility_entropy"] = summary["machine_eligibility_entropy"]
    metadata["bottleneck_coverage"] = summary["bottleneck_coverage"]
    metadata["bottleneck_intensity"] = summary["bottleneck_intensity"]
    metadata["processing_time_gap_ratio"] = summary["processing_time_gap_ratio"]
    metadata["time_skew"] = summary["time_skew"]
    metadata["params"]["post_processed"] = "machine 1 forced to ~0.3x fastest alternative"
    return instance, metadata


def machine_advantage_except_last_job(seed: int = EDGE_CASE_SEED):
    """
    Machine 1 is strongly advantageous on every job except the final job,
    where it is removed from eligibility entirely.
    """
    instance, metadata = generate_instance(
        number_of_jobs=10, number_of_machines=5,
        operations_per_job=(4, 8), machine_flexibility=0.5,
        processing_time_range=(10, 60), processing_time_variance=0.15,
        bottleneck_probability=0.5, seed=seed, flexibility_mode="random",
        bottleneck_machines=[1],
        instance_class_label="edge_machine_advantage_except_last_job",
    )

    last_job = len(instance["jobs"])
    for (j, k), eligible in list(instance["E"].items()):
        if j == last_job:
            if 1 in eligible:
                instance["E"][(j, k)] = [m for m in eligible if m != 1]
                instance["P"].pop((j, k, 1), None)
            continue

        if 1 in eligible:
            other_times = [
                instance["P"][(j, k, m)] for m in eligible if m != 1
            ]
            if other_times:
                instance["P"][(j, k, 1)] = max(1, round(0.3 * min(other_times)))

    assert_well_formed(instance)
    summary = summarize_instance(instance)
    metadata["computed_beta"] = summary["beta"]
    metadata["computed_dv"] = summary["dv"]
    metadata["machine_eligibility_entropy"] = summary["machine_eligibility_entropy"]
    metadata["bottleneck_coverage"] = summary["bottleneck_coverage"]
    metadata["bottleneck_intensity"] = summary["bottleneck_intensity"]
    metadata["processing_time_gap_ratio"] = summary["processing_time_gap_ratio"]
    metadata["time_skew"] = summary["time_skew"]
    metadata["params"]["post_processed"] = (
        "machine 1 made dominant on jobs 1..n-1, then removed from the final job"
    )
    return instance, metadata


def rigid_single_machine_huge_duration(seed: int = EDGE_CASE_SEED):
    """
    A single-machine instance whose durations are scaled up by a large factor.
    """
    instance, metadata = generate_instance(
        number_of_jobs=10, number_of_machines=1,
        operations_per_job=(4, 8), machine_flexibility=1.0,
        processing_time_range=(10, 60), processing_time_variance=0.15,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_rigid_single_machine_huge_duration",
    )

    scale = 1000
    for key in list(instance["P"]):
        instance["P"][key] = max(1, instance["P"][key] * scale)

    assert_well_formed(instance)
    summary = summarize_instance(instance)
    metadata["computed_beta"] = summary["beta"]
    metadata["computed_dv"] = summary["dv"]
    metadata["machine_eligibility_entropy"] = summary["machine_eligibility_entropy"]
    metadata["bottleneck_coverage"] = summary["bottleneck_coverage"]
    metadata["bottleneck_intensity"] = summary["bottleneck_intensity"]
    metadata["processing_time_gap_ratio"] = summary["processing_time_gap_ratio"]
    metadata["time_skew"] = summary["time_skew"]
    metadata["params"]["post_processed"] = "all durations scaled by 1000 on a rigid single-machine instance"
    return instance, metadata


def near_total_flexibility(seed: int = EDGE_CASE_SEED):
    """beta close to 1.0: almost every operation eligible on almost every machine."""
    return generate_instance(
        number_of_jobs=10, number_of_machines=8,
        operations_per_job=(4, 8), machine_flexibility=0.95,
        processing_time_range=(5, 50), processing_time_variance=0.2,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="total",
        instance_class_label="edge_near_total_flexibility",
    )


def near_zero_flexibility(seed: int = EDGE_CASE_SEED):
    """
    beta close to its minimum possible value (1/m): every operation has
    exactly one eligible machine.
    """
    m = 8
    return generate_instance(
        number_of_jobs=10, number_of_machines=m,
        operations_per_job=(4, 8), machine_flexibility=1.0 / m,
        processing_time_range=(5, 50), processing_time_variance=0.2,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_near_zero_flexibility",
    )


def identical_processing_times(seed: int = EDGE_CASE_SEED):
    """processing_time_variance=0: every eligible machine takes exactly the
    operation's base duration."""
    return generate_instance(
        number_of_jobs=10, number_of_machines=6,
        operations_per_job=(4, 8), machine_flexibility=0.5,
        processing_time_range=(5, 50), processing_time_variance=0.0,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_identical_processing_times",
    )


def long_critical_chain(seed: int = EDGE_CASE_SEED):
    """A single job with a very long operation chain,makespan is
    dominated by one unavoidable sequential path."""
    return generate_instance(
        number_of_jobs=1, number_of_machines=6,
        operations_per_job=(50, 50), machine_flexibility=0.4,
        processing_time_range=(5, 20), processing_time_variance=0.15,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_long_critical_chain",
    )


def many_short_operations(seed: int = EDGE_CASE_SEED):
    """Many jobs, few operations each, tiny processing times."""
    return generate_instance(
        number_of_jobs=60, number_of_machines=8,
        operations_per_job=(1, 2), machine_flexibility=0.4,
        processing_time_range=(1, 3), processing_time_variance=0.1,
        bottleneck_probability=0.0, seed=seed, flexibility_mode="random",
        instance_class_label="edge_many_short_operations",
    )


ALL_EDGE_CASES = {
    "single_machine": single_machine,
    "one_job": one_job,
    "single_bottleneck": single_bottleneck,
    "extreme_time_gap": extreme_time_gap,
    "machine_advantage": machine_advantage,
    "machine_advantage_except_last_job": machine_advantage_except_last_job,
    "rigid_single_machine_huge_duration": rigid_single_machine_huge_duration,
    "near_total_flexibility": near_total_flexibility,
    "near_zero_flexibility": near_zero_flexibility,
    "identical_processing_times": identical_processing_times,
    "long_critical_chain": long_critical_chain,
    "many_short_operations": many_short_operations,
}


def build_edge_case(name: str, seed: int = EDGE_CASE_SEED):
    if name not in ALL_EDGE_CASES:
        raise ValueError(
            f"unknown edge case '{name}'. Available: {list(ALL_EDGE_CASES)}"
        )
    return ALL_EDGE_CASES[name](seed=seed)
