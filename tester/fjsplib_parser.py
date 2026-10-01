"""
fjsplib_parser.py -- parses the standard FJSPLib text format (the one
used by Hurink, Brandimarte, DPpaulli, Kacem, Fattahi, Behnke-Geiger,
and every instance on https://scheduleopt.github.io/benchmarks/fjsplib/)
into this project's instance dict shape: {"jobs", "machines", "E", "P"}.

Format, in brief:
  Line 1:  n_jobs  n_machines  [optional avg_machines_per_op]
  Lines 2..n_jobs+1: one line per job, a flat sequence of integers:
      n_ops_in_job
      [count_1  (machine,time) x count_1]   <- operation 1
      [count_2  (machine,time) x count_2]   <- operation 2
      ...

"""

from pathlib import Path
from typing import Dict, List, Union


def parse_fjsplib_instance(text: str) -> dict:
    """
    Parses raw FJSPLib-format text (as a single string) into an
    instance dict: {"jobs", "machines", "E", "P"},the exact same
    shape generator.generate_instance() produces, so it plugs
    directly into everything else in this project (validator, WOA,
    LNS) with no special-casing.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]

    header = lines[0].split()
    n_jobs = int(header[0])
    n_machines = int(header[1])
    # header[2], if present, is just "average machines per operation" --
    # informational metadata. We don't use it; generator.metrics.compute_beta
    # recomputes the same thing from the parsed data as a cross-check.

    jobs: Dict[int, List[int]] = {}
    E: Dict[tuple, List[int]] = {}
    P: Dict[tuple, int] = {}

    for job_id, line in enumerate(lines[1:1 + n_jobs], start=1):
        tokens = list(map(int, line.split()))
        pos = 0

        n_ops = tokens[pos]; pos += 1
        jobs[job_id] = list(range(1, n_ops + 1))

        for op_id in range(1, n_ops + 1):
            count = tokens[pos]; pos += 1
            eligible = []
            for _ in range(count):
                machine_id = tokens[pos]; pos += 1
                proc_time = tokens[pos]; pos += 1
                eligible.append(machine_id)
                P[(job_id, op_id, machine_id)] = proc_time
            E[(job_id, op_id)] = eligible

        if pos != len(tokens):
            raise ValueError(
                f"job {job_id}: token count mismatch -- consumed {pos} of "
                f"{len(tokens)} tokens. The file may not be in FJSPLib format, "
                f"or this job's line is malformed."
            )

    return {
        "jobs": jobs,
        "machines": list(range(1, n_machines + 1)),
        "E": E,
        "P": P,
    }


def parse_fjsplib_file(path: Union[str, Path]) -> dict:
    """Convenience wrapper: read a .txt file and parse it."""
    with open(path, "r") as f:
        return parse_fjsplib_instance(f.read())


def load_fjsplib_folder(folder: Union[str, Path], pattern: str = "*.txt") -> Dict[str, dict]:
    """
    Loads every FJSPLib-format file in a folder (e.g. a cloned copy of
    https://github.com/ScheduleOpt/benchmarks 's instances/fjsp folder).
    Returns {filename_without_extension: instance_dict}.
    """
    folder = Path(folder)
    instances = {}
    for filepath in sorted(folder.glob(pattern)):
        instances[filepath.stem] = parse_fjsplib_file(filepath)
    return instances
