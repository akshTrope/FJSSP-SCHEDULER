
from .instance_generator import generate_instance

INSTANCE_CLASS_PRESETS = {
    "average": dict(
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.30,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=0.05, flexibility_mode="random",
    ),
    "low_flexibility": dict(
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.10,
        processing_time_range=(5, 50), processing_time_variance=0.10,
        bottleneck_probability=0.05, flexibility_mode="close",
    ),
    "high_flexibility": dict(
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.80,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=0.05, flexibility_mode="total",
    ),
    "bottleneck_heavy": dict(
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.30,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=0.65, flexibility_mode="random",
    ),
    "high_variance": dict(
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.30,
        processing_time_range=(5, 50), processing_time_variance=0.70,
        bottleneck_probability=0.05, flexibility_mode="random",
    ),
    "unbalanced_ratio": dict(
        #16.7 jobs per machine 
        number_of_jobs=50, number_of_machines=3,
        operations_per_job=(3, 6), machine_flexibility=0.30,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=0.05, flexibility_mode="random",
    ),
    "extreme": dict(
        #extreme instance ->400+ operations, 40+ jobs, 50+ machines, low flexibility, high variance, high bottleneck probability
        number_of_jobs=40, number_of_machines=6,
        operations_per_job=(8, 15), machine_flexibility=0.10,
        processing_time_range=(1, 200), processing_time_variance=0.80,
        bottleneck_probability=0.70, flexibility_mode="close",
    ),
    "skewed_durations": dict(
        # processing_time_distribution="lognormal": most
        # operations end up short, with a long right tail of a few very
        # long ones, instead of the flat Uniform(pt_min, pt_max) base.
        number_of_jobs=15, number_of_machines=10,
        operations_per_job=(5, 10), machine_flexibility=0.30,
        processing_time_range=(5, 50), processing_time_variance=0.15,
        bottleneck_probability=0.05, flexibility_mode="random",
        processing_time_distribution="lognormal",
    ),
    "rand01": dict(
        number_of_jobs=30, number_of_machines=18,
        operations_per_job=(5, 20), machine_flexibility=0.6,
        processing_time_range=(5, 100), processing_time_variance=0.50,
        bottleneck_probability=0.20, flexibility_mode="random",
        processing_time_distribution="lognormal",
        ),
    "rand02": dict(
        #Pretty Huge Instance (Expect Cp-SAT to fail in the given time)
        number_of_jobs=45, number_of_machines=27,
        operations_per_job=(2, 60), machine_flexibility=0.3,
        processing_time_range=(5, 25), processing_time_variance=0.70,
        bottleneck_probability=0.60, flexibility_mode="close",
        processing_time_distribution="normal",
        ),
    "rand03": dict(
        number_of_jobs=30, number_of_machines=10,
        operations_per_job=(5, 12), machine_flexibility=0.3,
        processing_time_range=(5, 50), processing_time_variance=0.30,
        bottleneck_probability=1.0, flexibility_mode="random",
        processing_time_distribution="normal",bottleneck_stage_filter="all",
        bottleneck_machines=[1,2,3,4,5,6],
        ),
    
}

def generate_from_preset(class_name: str, seed: int, overrides: dict = None):
    """
    Convenience wrapper: generate_instance() using a named preset, with
    optional parameter overrides layered on top.
    """
    if class_name not in INSTANCE_CLASS_PRESETS:
        raise ValueError(
            f"unknown instance class '{class_name}'. "
            f"Available: {list(INSTANCE_CLASS_PRESETS)}"
        )
    params = dict(INSTANCE_CLASS_PRESETS[class_name])
    if overrides:
        params.update(overrides)
    return generate_instance(seed=seed, instance_class_label=class_name, **params)

