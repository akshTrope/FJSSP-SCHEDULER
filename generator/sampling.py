"""
Sampling primitives used by the generator.

Every function here is pure (given the same `rng` state, returns the same
result) so the whole generator is reproducible from a single seed.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

# can be an integer/tuple
IntOrRange = Union[int, Tuple[int, int]]
# a bottleneck entry is either a bare machine id (uses the default speed
# factor) or an (machine_id, (lo, hi)) pair with its own speed range
BottleneckEntry = Union[int, Tuple[int, Tuple[float, float]]]

# Default noise band used for "close" flexibility mode, machines in the same window get processing times
# that stay within this fraction of each other.
DEFAULT_CLOSE_MODE_VARIANCE = 0.05

# For "normal"/"lognormal" processing_time_distribution: sigma is derived
# from the requested (pt_min, pt_max) range so that range covers roughly
# +/-3 standard deviations.
DEFAULT_SIGMA_DIVISOR = 6.0

#(How big is the long and short region)
DEFAULT_BIMODAL_EDGE_FRACTION = 0.2

#(How often we choose the long region)
DEFAULT_BIMODAL_LONG_FRACTION = 0.2


def resolve_count(spec: IntOrRange, rng) -> int:
    """
    spec is either a fixed int, or an inclusive (min, max) tuple sampled
    uniformly. Used for number_of_jobs and operations_per_job.
    """
    if isinstance(spec, tuple):
        lo, hi = spec
        return rng.randint(lo, hi)
    return spec


def resolve_job_length(
    operations_per_job: IntOrRange,
    job_length_mode: str,
    rng,
    bimodal_long_fraction: float = DEFAULT_BIMODAL_LONG_FRACTION,
    bimodal_edge_fraction: float = DEFAULT_BIMODAL_EDGE_FRACTION,
) -> int:
    """
    Resolves how many operations a single job gets.

    job_length_mode="uniform" (default): identical to plain resolve_count
    -- a fixed int, or a uniform draw over the given range.

    job_length_mode="bimodal": jobs are pulled from two clusters near the
    two ends of the operations_per_job range -- a "short" cluster near
    the low end and a "long" cluster near the high end -- instead of
    filling the range evenly. bimodal_long_fraction controls how many
    jobs land in the long cluster (default 0.2 -> a few long jobs, many
    short ones).
    bimodal_edge_fraction controls how wide each cluster is, as a
    fraction of the full range. Falls back to resolve_count if
    operations_per_job is a fixed int (nothing to split across).
    """
    if job_length_mode == "uniform":
        return resolve_count(operations_per_job, rng)

    if job_length_mode == "bimodal":
        if not isinstance(operations_per_job, tuple):
            return resolve_count(operations_per_job, rng)
        lo, hi = operations_per_job
        span = hi - lo
        edge = max(0, round(span * bimodal_edge_fraction))
        short_hi = min(hi, lo + edge)
        long_lo = max(lo, hi - edge)
        if rng.random() < bimodal_long_fraction:
            return rng.randint(long_lo, hi)
        return rng.randint(lo, short_hi)

    raise ValueError(f"unknown job_length_mode: {job_length_mode!r}")


def sample_eligible_machines(
    machines: List[int],
    target_beta: float,
    mode: str,
    rng,
) -> List[int]:
    """
    Returns the eligible machine set E(j,k) for a single operation.

    target_beta in [0, 1] sets the size of the set: flex_count =
    round(target_beta * m), clipped to [1, m] so an operation always has
    at least one eligible machine (well-formedness rule).

    mode controls which machines are chosen:
      "random" (rdata-like) : uniform sample, no locality -> scattered set
      "close"  (edata-like (modified for time similarity as well)) : contiguous window around a random anchor
                               -> clustered set of "similar" machines
      "total"  (vdata-like) : forces flex_count up near m, ignoring
                               target_beta -> near-full flexibility
    """
    m = len(machines)
    flex_count = round(target_beta * m)
    flex_count = max(1, min(flex_count, m))

    if mode == "total":
        flex_count = max(flex_count, round(0.9 * m))
        flex_count = min(flex_count, m)
        #sample a subset of machines of size flex_count from the list of machines
        return rng.sample(machines, flex_count)

    if mode == "close":
        anchor = rng.choice(machines)
        anchor_idx = machines.index(anchor)
        window = [
            machines[(anchor_idx + offset) % m]
            for offset in range(flex_count)
        ]
        return list(dict.fromkeys(window))  #preserve order

    if mode == "random":
        return rng.sample(machines, flex_count)

    raise ValueError(f"unknown flexibility_mode: {mode!r}")


def sample_base_duration(
    pt_min: int,
    pt_max: int,
    distribution: str,
    rng,
    sigma_divisor: float = DEFAULT_SIGMA_DIVISOR,
) -> int:
    """
    Samples a single "base" duration for one operation.

      "uniform"   (default): Uniform(pt_min, pt_max).
      "normal"    : Gaussian centered at the midpoint of the range, with
                    sigma = (pt_max - pt_min) / sigma_divisor so the
                    requested range covers roughly +/-3 standard
                    deviations by default clipped back into
                    [pt_min, pt_max] so the generator never
                    produces a duration outside what was asked for.
      "lognormal" : the same +/-3-sigma logic, but computed in log-space,
                    so most operations come out short with a long right
                    tail of a few very long ones.

    Always returns a strictly positive integer.
    """
    if distribution == "uniform":
        base = rng.randint(pt_min, pt_max)
    #useful for similar duration jobs
    elif distribution == "normal":
        mu = (pt_min + pt_max) / 2.0
        sigma = max((pt_max - pt_min) / sigma_divisor, 1e-9)
        base = round(rng.gauss(mu, sigma))
        base = min(max(base, pt_min), pt_max)
    #useful for many short tasks and a very few long tasks.
    elif distribution == "lognormal":
        # log(0) is undefined, so floor pt_min at 1 for the log-space math
        # the returned value is still clipped into [pt_min, pt_max].
        safe_min = max(pt_min, 1)
        safe_max = max(pt_max, safe_min + 1)
        log_lo, log_hi = math.log(safe_min), math.log(safe_max)
        mu_log = (log_lo + log_hi) / 2.0
        sigma_log = max((log_hi - log_lo) / sigma_divisor, 1e-9)
        base = round(math.exp(rng.gauss(mu_log, sigma_log)))
        base = min(max(base, pt_min), pt_max)

    else:
        raise ValueError(f"unknown processing_time_distribution: {distribution!r}")

    return max(1, base)


def normalize_bottleneck_config(
    #Only machine id's or (machine_id, (lo, hi)) pairs are allowed
    bottleneck_machines: Optional[List[BottleneckEntry]],
    default_speed_factor: Tuple[float, float],
) -> Dict[int, Tuple[float, float]]:
    """
    Normalizes the bottleneck_machines argument into
    {machine_id: (speed_lo, speed_hi)}.

    Accepts a mix of:
      - bare machine ids, e.g. [1, 3]        -> each uses default_speed_factor
      - (machine_id, (lo, hi)) pairs         -> explicit, independently
                                                 tunable speed range per
                                                 machine, e.g.
                                                 [(1, (0.5, 0.8)), (3, (0.2, 0.3))]
    None defaults to a single bottleneck on machine 1, matching the
    generator's original default behavior.
    """
    if bottleneck_machines is None:
        bottleneck_machines = [1]

    speed_map: Dict[int, Tuple[float, float]] = {}
    for entry in bottleneck_machines:
        if isinstance(entry, tuple):
            machine_id, speed_range = entry
        else:
            machine_id, speed_range = entry, default_speed_factor
        speed_map[machine_id] = speed_range
    return speed_map

#which machine is a candidate for bottleneck injection, based on the stage of the operation in its job
def bottleneck_stage_matches(k: int, k_total: int, stage_filter: str) -> bool:
    """
    Whether operation index k (1-indexed, out of k_total in its job) is a
    candidate for bottleneck injection under stage_filter:

      "all"    : every operation is a candidate (original behavior)
      "first"  : only each job's first operation
      "last"   : only each job's last operation
      "middle" : only operations that are neither first nor last
                 (a job with fewer than 3 operations has no middle)
    """
    if stage_filter == "all":
        return True
    if stage_filter == "first":
        return k == 1
    if stage_filter == "last":
        return k == k_total
    if stage_filter == "middle":
        return 1 < k < k_total

    raise ValueError(f"unknown bottleneck_stage_filter: {stage_filter!r}")


def sample_processing_times(
    eligible_machines: List[int],
    pt_min: int,
    pt_max: int,
    processing_time_variance: float,
    flexibility_mode: str,
    bottleneck_speed_map: Dict[int, Tuple[float, float]],
    rng,
    distribution: str = "uniform",
    close_mode_variance: float = DEFAULT_CLOSE_MODE_VARIANCE,
) -> Dict[int, int]:
    """
    Returns {machine_id: processing_time} for one operation, covering
    exactly the machines in `eligible_machines` (never more, never fewer).

    base is sampled once per operation via sample_base_duration() (shape
    controlled by `distribution`). Each eligible machine's time =
    base * Uniform(1-variance, 1+variance), rounded and clamped to be
    strictly positive.

    If flexibility_mode == "close", the noise band is tightened to
    close_mode_variance (regardless of the requested variance) so that
    machines within the same "close" window end up nearly interchangeable
    in duration machines in a "close" window are meant to represent
    similar/nearby equipment, not just eligibility-wise but time-wise too.

    Any machine present in bottleneck_speed_map additionally gets
    multiplied by that machine's own (lo, hi) speed factor range,
    independently of every other bottleneck machine,this is applied
    after the mode-based noise, independent of it, and applies wherever
    that machine happens to be eligible (whether it was forcibly
    injected into E(j,k) or was already eligible from the base flexible
    sampling step).
    """
    base = sample_base_duration(pt_min, pt_max, distribution, rng)

    if flexibility_mode == "close":
        effective_variance = min(processing_time_variance, close_mode_variance)
    else:
        effective_variance = processing_time_variance

    noise_lo = 1 - effective_variance
    noise_hi = 1 + effective_variance

    times = {}
    for mach in eligible_machines:
        factor = rng.uniform(noise_lo, noise_hi)
        if mach in bottleneck_speed_map:
            factor *= rng.uniform(*bottleneck_speed_map[mach])

        p_val = round(base * factor)
        p_val = max(1, p_val)  # strictly positive
        times[mach] = p_val

    return times
