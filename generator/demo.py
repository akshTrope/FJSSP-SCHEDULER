import sys
from pathlib import Path

from .presets import INSTANCE_CLASS_PRESETS, generate_from_preset
from .edge_cases import ALL_EDGE_CASES, build_edge_case
from .io_utils import save_instance

DEFAULT_SEED = 39
INSTANCES_DIR = Path(__file__).resolve().parent.parent / "instances"
flag = False


def main():
    rows = []

    for class_name in INSTANCE_CLASS_PRESETS:
        instance, metadata = generate_from_preset(class_name, seed=DEFAULT_SEED)
        out_path = INSTANCES_DIR / f"{class_name}_seed{DEFAULT_SEED}.json"
        save_instance(instance, metadata, str(out_path))
        rows.append((
            class_name, metadata["n_jobs"], metadata["n_machines"],
            metadata["n_operations"], metadata["computed_beta"],
            metadata["computed_dv"], metadata["machine_eligibility_entropy"],
            metadata["bottleneck_coverage"], metadata["bottleneck_intensity"],
            metadata["processing_time_gap_ratio"], metadata["time_skew"],
        ))

    for edge_name in ALL_EDGE_CASES:
        instance, metadata = build_edge_case(edge_name)
        out_path = INSTANCES_DIR / f"{edge_name}.json"
        save_instance(instance, metadata, str(out_path))
        rows.append((
            edge_name, metadata["n_jobs"], metadata["n_machines"],
            metadata["n_operations"], metadata["computed_beta"],
            metadata["computed_dv"], metadata["machine_eligibility_entropy"],
            metadata["bottleneck_coverage"], metadata["bottleneck_intensity"],
            metadata["processing_time_gap_ratio"], metadata["time_skew"],
        ))

    header = (
        f"{'instance':30s} {'n_jobs':>7s} {'n_mach':>7s} {'n_ops':>7s} "
        f"{'beta':>8s} {'dv':>8s} {'entropy':>9s} {'cov':>8s} {'int':>8s} "
        f"{'gap':>8s} {'skew':>8s}"
    )

    if flag == True:
        print(header)
        print("-" * len(header))

        for row in rows:
            name, n_jobs, n_mach, n_ops, beta, dv, entropy, cov, intensity, gap, skew = row

            print(
                f"{name:30s} {n_jobs:7d} {n_mach:7d} {n_ops:7d} "
                f"{beta:8.3f} {dv:8.3f} {entropy:9.3f} {cov:8.3f} {intensity:8.3f} "
                f"{gap:8.3f} {skew:8.3f}"
            )

        print(f"\nSaved {len(rows)} instance files to {INSTANCES_DIR}")


if __name__ == "__main__":
    main()