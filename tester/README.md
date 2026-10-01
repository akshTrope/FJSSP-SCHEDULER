# Tester

The tester runs an FJSP instance through the project's current comparison algorithm: CP-SAT first, followed by the WOA + LNS hybrid when needed. It validates the schedules and reports the selected result, runtime, instance metrics, and BKS gap when a published bound is available.

Run commands from the project root (`fjsp/`).

## Requirements

Use Python 3 with NumPy and OR-Tools installed. From the project root, install them if needed:

```bash
python3 -m pip install numpy ortools
```

## Run One Benchmark

Pass the benchmark's name without the `.txt` extension:

```bash
python3 -m tester.run_compare mk01
python3 -m tester.run_compare behnke1
```

If a name exists in several benchmark variants, the runner reports that it is ambiguous. Hurink instances commonly have `edata`, `rdata`, `sdata`, and `vdata` versions. Run a specific variant by giving its full path:

```bash
python3 -m tester.run_compare --path benchmarks/flexible-jobshop/instances/fjsp/HurinkJurischThole1994/edata/la33.txt
```

## Run A Generated Instance

The tester accepts JSON files saved by the generator, including their metadata:

```bash
python3 -m tester.run_compare --path instances/skewed_durations_seed39.json
```

It also accepts a single FJSPLib `.txt` instance using `--path`.

## Run A Folder

Pass a folder containing `.txt` or `.json` instance files. The runner processes files directly inside that folder; it does not search nested folders.

```bash
python3 -m tester.run_compare --path benchmarks/flexible-jobshop/instances/fjsp/HurinkJurischThole1994/edata
```

To run all instances in a family folder:

```bash
python3 -m tester.run_compare --path benchmarks/flexible-jobshop/instances/fjsp/Brandimarte1993
```

## How The Comparison Works

1. CP-SAT runs once with a 10-second limit and 8 search workers.
2. If CP-SAT returns a valid `OPTIMAL` schedule, that result is selected and the hybrid is skipped.
3. Otherwise, the WOA + LNS hybrid runs with seeds `0`, `1`, `2`, and `3`. Each seed has a 15-second total limit; each LNS neighborhood's CP-SAT solve is capped at 3 seconds within that limit.
4. The best valid hybrid result is selected. If CP-SAT found a valid feasible schedule, that schedule is retained when the hybrid does not improve its makespan. If CP-SAT found no usable schedule, the best valid hybrid result is used.

When the hybrid runs, the reported runtime is the total elapsed comparison time, including the initial CP-SAT run and all four hybrid seeds. The maximum is approximately 70 seconds, plus small overhead. If CP-SAT proves optimal immediately, the hybrid does not run.

The terminal output lists each hybrid seed's makespan and identifies the best seed. The selected schedule is printed only when `PRINT_SCHEDULE` is enabled in `tester/run_compare.py`.

## Results

By default, each completed comparison is appended to:

```text
results/my_instance_results.csv
```

The CSV contains the instance name and path, structural metrics, selected method, validation status, makespan, total runtime, published lower and upper bounds, and one gap percentage to the nearer bound when BKS data exists. Generated instances usually have no BKS entry, so their gap is shown as unavailable.

Choose another CSV path with `--table`, or prevent saving with `--no-save`:

```bash
python3 -m tester.run_compare mk01 --table results/brandimarte_results.csv
python3 -m tester.run_compare mk01 --no-save
```

View saved results without running solvers:

```bash
python3 -m tester.run_compare --view-table
python3 -m tester.run_compare --view-table --last 10
```

## Useful Paths

### Needed by `run_compare.py`

- `tester/run_compare.py`: main command-line comparison runner and solver settings
- `tester/best_known_results.py`: benchmark BKS lookup and gap calculation
- `tester/fjsplib_parser.py`: parser for FJSPLib `.txt` instances
- `tester/results_table.py`: CSV output and results-table display
- `tester/__init__.py`: package marker, required when using `python3 -m tester.run_compare`

The runner also needs the project-level `algorithms/` and `generator/` modules,
one input instance (`benchmarks/.../*.txt` or `instances/*.json`), and
`benchmarks/flexible-jobshop/solutions/bks.json` when reporting benchmark gaps.