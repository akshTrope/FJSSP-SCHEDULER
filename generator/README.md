# FJSP Instance Generator

This folder contains the **Flexible Job Shop Scheduling Problem (FJSP) instance generator** used to create randomized and hand-crafted benchmark instances.

The generator supports configurable job sizes, operation counts, machine flexibility, processing times, bottleneck machines, and other instance characteristics. It also provides utilities for computing instance metrics and validating generated schedules.

---

## Directory Structure

The `generator/` package contains the following modules:

| File                    | Description                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sampling.py`           | Provides the core sampling functions for generating base processing times, number of jobs and operations, and eligible machine sets for each operation. |
| `metrics.py`            | Defines metrics used to analyze and characterize generated FJSP instances.                                                                              |
| `instance_generator.py` | Uses the sampling functions to generate complete FJSP instances. It also computes instance metrics using the utilities from `metrics.py`.               |
| `edge_cases.py`         | Contains hand-crafted FJSP edge cases designed to test scheduling algorithms against specific scenarios.                                                |
| `validation.py`         | Validates generated schedules according to the FJSP problem constraints.                                                                                |
| `presets.py`            | Contains predefined parameter configurations for different named instance classes.                                                                      |
| `demo.py`               | Demonstrates instance generation using the predefined instance classes and edge cases.                                                                  |

---
## 5. Instance characteristics that matter (and why)

- **`beta`** — average fraction of machines each operation is eligible for.
- **`dv` (duration variety)** — how many distinct processing-time values exist relative to how many machine-choice options exist.
- **`processing_time_gap_ratio` / `time_skew`** — how much a wrong machine choice actually costs. Moderate beta *combined with* a meaningful gap ratio (few choices, each with real consequences) creates a situation that's harder for both WOA's continuous moves and CP-SAT's branching to navigate cleanly.
- **`bottleneck_coverage` / `bottleneck_intensity`** — how widely relied-upon, and how meaningfully different in speed, a bottleneck machine is. Both need to be notable together for a real hardness effect; either alone is not sufficient.


## Generating Instances

Run the following command from the `fjsp/` directory:

```bash
python -m generator.demo
```

This generates:

* One instance for each named instance class defined in `INSTANCE_CLASS_PRESETS`.
* All hand-built edge-case instances defined in `edge_cases.py`.

The generated instances are written to:

```text
../instances/
```

---

## Instance Class Presets

Predefined instance classes and their parameter configurations are defined in:

```text
generator/presets.py
```

Specifically, the `INSTANCE_CLASS_PRESETS` dictionary contains the parameter presets used to generate each named instance class.

These presets are based on the instance-analysis metrics defined in `metrics.py`.

---

## Custom Instance Generation

A custom FJSP instance can be generated directly using `generate_instance()`.

### Example

```python
from pathlib import Path

from generator import generate_instance, save_instance

seed = 123

instance, metadata = generate_instance(
    number_of_jobs=8,
    number_of_machines=5,
    operations_per_job=(4, 7),
    machine_flexibility=0.4,
    processing_time_range=(5, 40),
    processing_time_variance=0.2,
    bottleneck_probability=0.1,
    bottleneck_machines=[1],
    bottleneck_speed_factor=(0.5, 0.8),
    flexibility_mode="random",
    instance_class_label="my_custom",
    seed=seed,
)

path = Path("instances") / f"my_custom_seed{seed}.json"

save_instance(instance, metadata, str(path))

print(metadata)
```

This creates an instance with:

* **8 jobs**
* **5 machines**
* **4–7 operations per job**
* **40% machine flexibility**
* Processing times between **5 and 40**
* Processing-time variance of **0.2**
* **10% bottleneck probability**
* Machine `1` designated as a bottleneck machine
* Bottleneck speed factor between **0.5 and 0.8**
* Random machine-flexibility assignment
* Instance label: `my_custom`
* Reproducible generation using seed `123`

The resulting instance is saved as:

```text
instances/my_custom_seed123.json
```

---

## Main Generation Parameters

| Parameter                  | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `number_of_jobs`           | Number of jobs in the generated FJSP instance.                            |
| `number_of_machines`       | Number of available machines.                                             |
| `operations_per_job`       | Range `(min, max)` specifying the number of operations for each job.      |
| `machine_flexibility`      | Controls how many machines are eligible to process each operation.        |
| `processing_time_range`    | Minimum and maximum base processing time.                                 |
| `processing_time_variance` | Controls variation in sampled processing times.                           |
| `bottleneck_probability`   | Probability of introducing bottleneck-related processing characteristics. |
| `bottleneck_machines`      | Machines designated as potential bottlenecks.                             |
| `bottleneck_speed_factor`  | Range of processing-speed factors applied to bottleneck machines.         |
| `flexibility_mode`         | Determines how eligible machines are selected for operations.             |
| `instance_class_label`     | Label assigned to the generated instance class.                           |
| `seed`                     | Random seed used to make instance generation reproducible.                |

---

## Schedule Validation

The `validation.py` module provides functionality for checking whether a generated schedule satisfies the constraints of the FJSP problem.

Validation can be used to verify properties such as:

* Operation precedence within each job
* Machine eligibility
* Machine non-overlap
* Processing durations
* Overall schedule feasibility

This makes the validation module useful for testing schedules produced by different scheduling algorithms.

---

## Edge Cases


The `edge_cases.py` module contains manually constructed instances designed to test scheduling algorithms under specific conditions.

---

## Reproducibility

Instance generation is controlled using a random seed:

```python
seed = 123
```

Using the same generation parameters and seed produces the same instance, making experiments reproducible.