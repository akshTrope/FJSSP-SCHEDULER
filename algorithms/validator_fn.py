"""Independent validation for solver-produced FJSP schedules."""

from typing import Any, Dict, List

from generator.validation import InstanceValidationError, assert_well_formed


def check(
    instance: Dict[str, Any],
    job_id: int,
    operation_id: int,
    machine_id: int,
    start_time: int,
    completion_time: int,
) -> bool:
    """Return whether one schedule entry is valid for the instance."""
    try:
        _validate_entry(
            instance,
            {
                "job_id": job_id,
                "operation_id": operation_id,
                "machine_id": machine_id,
                "start_time": start_time,
                "completion_time": completion_time,
            },
            1,
        )
    except InstanceValidationError:
        return False
    return True


def validate_schedule(instance: Dict[str, Any], schedule: Any) -> Dict[str, Any]:
    """Validate a complete schedule and return normalized summaries."""
    assert_well_formed(instance)
    normalized = _normalize_schedule(schedule)
    jobs = instance["jobs"]
    machines = set(instance["machines"])
    expected_count = sum(len(operation_ids) for operation_ids in jobs.values())

    if len(normalized) != expected_count:
        raise InstanceValidationError(
            f"Schedule size is incorrect: expected {expected_count} scheduled "
            f"operations, got {len(normalized)}."
        )

    seen = set()
    job_schedules = {job_id: [] for job_id in jobs}
    machine_schedules = {machine_id: [] for machine_id in machines}

    for index, entry in enumerate(normalized, 1):
        _validate_entry(instance, entry, index)
        operation = (entry["job_id"], entry["operation_id"])
        if operation in seen:
            raise InstanceValidationError(
                f"Duplicate operation entry for O({operation[0]},{operation[1]})."
            )
        seen.add(operation)
        job_schedules[entry["job_id"]].append(entry)
        machine_schedules[entry["machine_id"]].append(entry)

    expected_operations = {
        (job_id, operation_id)
        for job_id, operation_ids in jobs.items()
        for operation_id in operation_ids
    }
    if seen != expected_operations:
        missing = sorted(expected_operations - seen)
        extra = sorted(seen - expected_operations)
        raise InstanceValidationError(
            f"Schedule operations do not match the instance. Missing: {missing}; extra: {extra}."
        )

    for job_id, entries in job_schedules.items():
        ordered = sorted(entries, key=lambda entry: entry["operation_id"])
        for previous, current in zip(ordered, ordered[1:]):
            if current["start_time"] < previous["completion_time"]:
                raise InstanceValidationError(
                    f"Precedence violation in job {job_id}: operation "
                    f"{current['operation_id']} starts before its predecessor finishes."
                )

    for machine_id, entries in machine_schedules.items():
        ordered = sorted(entries, key=lambda entry: (entry["start_time"], entry["completion_time"]))
        for previous, current in zip(ordered, ordered[1:]):
            if current["start_time"] < previous["completion_time"]:
                raise InstanceValidationError(
                    f"Machine {machine_id} has overlapping operations: "
                    f"O({previous['job_id']},{previous['operation_id']}) and "
                    f"O({current['job_id']},{current['operation_id']})."
                )

    job_completion_times = {
        job_id: max(entry["completion_time"] for entry in entries)
        for job_id, entries in job_schedules.items()
    }
    makespan = max(job_completion_times.values()) if job_completion_times else 0

    return {
        "normalized_schedule": normalized,
        "machine_schedules": {
            machine_id: sorted(
                entries,
                key=lambda entry: (entry["start_time"], entry["completion_time"]),
            )
            for machine_id, entries in sorted(machine_schedules.items())
        },
        "job_schedules": {
            job_id: sorted(entries, key=lambda entry: entry["operation_id"])
            for job_id, entries in sorted(job_schedules.items())
        },
        "job_completion_times": job_completion_times,
        "makespan": makespan,
    }


def _normalize_schedule(schedule: Any) -> List[Dict[str, Any]]:
    if isinstance(schedule, dict) and "schedule" in schedule:
        schedule = schedule["schedule"]
    if not isinstance(schedule, list):
        raise InstanceValidationError("Schedule must be provided as a list of entries.")

    normalized = []
    for index, entry in enumerate(schedule, 1):
        if isinstance(entry, (tuple, list)):
            if len(entry) != 5:
                raise InstanceValidationError(
                    f"Schedule entry #{index} must contain exactly five values."
                )
            fields = zip(
                ("job_id", "operation_id", "machine_id", "start_time", "completion_time"),
                entry,
            )
            normalized.append(dict(fields))
        elif isinstance(entry, dict):
            normalized.append(_extract_fields(entry, index))
        else:
            raise InstanceValidationError(
                f"Schedule entry #{index} must be a dictionary or 5-item tuple/list."
            )
    return normalized


def _extract_fields(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    aliases = {
        "job_id": ("job_id", "job", "j"),
        "operation_id": ("operation_id", "op_id", "operation", "k"),
        "machine_id": ("machine_id", "machine", "m"),
        "start_time": ("start_time", "start", "s"),
        "completion_time": ("completion_time", "completion", "finish_time", "end_time", "c"),
    }
    result = {}
    for field, names in aliases.items():
        for name in names:
            if name in entry:
                result[field] = entry[name]
                break
        else:
            raise InstanceValidationError(
                f"Schedule entry #{index} is missing the '{field}' field."
            )
    return result


def _validate_entry(instance: Dict[str, Any], entry: Dict[str, Any], index: int) -> None:
    job_id = entry["job_id"]
    operation_id = entry["operation_id"]
    machine_id = entry["machine_id"]
    start_time = entry["start_time"]
    completion_time = entry["completion_time"]

    if not isinstance(job_id, int) or job_id < 1 or job_id not in instance["jobs"]:
        raise InstanceValidationError(f"Schedule entry {index} has an invalid job_id: {job_id!r}.")
    if not isinstance(operation_id, int) or operation_id not in instance["jobs"][job_id]:
        raise InstanceValidationError(
            f"Schedule entry {index} references invalid operation O({job_id},{operation_id})."
        )
    if not isinstance(machine_id, int) or machine_id not in instance["machines"]:
        raise InstanceValidationError(f"Schedule entry {index} has an invalid machine_id: {machine_id!r}.")

    operation = (job_id, operation_id)
    if machine_id not in instance["E"][operation]:
        raise InstanceValidationError(
            f"Operation O({job_id},{operation_id}) is not eligible for machine {machine_id}."
        )
    if not isinstance(start_time, (int, float)) or not isinstance(completion_time, (int, float)):
        raise InstanceValidationError(f"Schedule entry {index} must use numeric times.")
    if start_time < 0 or completion_time <= start_time:
        raise InstanceValidationError(f"Schedule entry {index} has an invalid time interval.")

    expected_duration = instance["P"][(job_id, operation_id, machine_id)]
    if completion_time - start_time != expected_duration:
        raise InstanceValidationError(
            f"Schedule entry {index} has the wrong processing time for "
            f"O({job_id},{operation_id}) on machine {machine_id}. "
            f"Expected {expected_duration}, got {completion_time - start_time}."
        )
