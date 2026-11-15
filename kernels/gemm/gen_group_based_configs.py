#!/usr/bin/env python3
"""Generate a GEMM configuration CSV with a fixed number of configs per workload.

By default this selects 10 distinct (M, N, K) workloads from the configured
shape space and generates 200 unique kernel-parameter configurations for each
workload using discrete Latin Hypercube Sampling (LHS).

The output columns match the input expected by the user's existing GEMM
benchmark/evaluation script:
    M, N, K, TILE_M, TILE_N, TILE_K, THREAD_X, THREAD_Y, THREAD_Z,
    UNROLL_FACTOR

Example:
    python generate_gemm_grouped_configs.py
    python generate_gemm_grouped_configs.py --groups 10 --configs-per-group 200
    python generate_gemm_grouped_configs.py --seed 123 --output gemm_configurations.csv

Optional: choose workload shapes from an existing CSV (for example, a held-out
workload list) instead of selecting them from the built-in shape grid:
    python generate_gemm_grouped_configs.py --workloads-csv heldout_workloads.csv

The optional workloads CSV must contain M, N, and K columns. Its unique
(M, N, K) triples are sampled without replacement.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from scipy.stats import qmc


# Keep these values synchronized with the GEMM configuration space used by
# the original generator.
SHAPE_SPACE: Dict[str, List[int]] = {
    "M": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 14336],
    "N": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 14336],
    "K": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192],
}

KERNEL_PARAM_SPACE: Dict[str, List[int]] = {
    "TILE_M": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_N": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_K": [2, 4, 8, 16, 32, 64],
    "THREAD_X": [2, 4, 8, 16, 32, 64, 128, 256, 512],
    "THREAD_Y": [2, 4, 8, 16, 32, 64, 128, 256, 512],
    "THREAD_Z": [1],
    "UNROLL_FACTOR": [1, 2, 4, 8, 16, 32, 64],
}

GROUP_COLUMNS = ["M", "N", "K"]
FIELDNAMES = GROUP_COLUMNS + list(KERNEL_PARAM_SPACE.keys())
Workload = Tuple[int, int, int]


def discrete_lhs_samples(
    space: Dict[str, Sequence[int]],
    num_samples: int,
    seed: int,
) -> List[Dict[str, int]]:
    """Generate exactly num_samples unique discrete samples using LHS.

    LHS samples are generated in [0, 1)^d and mapped to each parameter's
    discrete choices. Samples that collapse to an already-seen discrete
    configuration are discarded and replaced with additional LHS points.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    keys = list(space)
    total_combinations = math.prod(len(space[key]) for key in keys)
    if num_samples > total_combinations:
        raise ValueError(
            f"Requested {num_samples} unique configurations, but the parameter "
            f"space contains only {total_combinations} combinations."
        )

    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    selected: List[Dict[str, int]] = []
    seen = set()

    while len(selected) < num_samples:
        remaining = num_samples - len(selected)
        # Oversampling reduces the number of LHS batches needed after discrete
        # mapping creates duplicate combinations.
        batch_size = max(remaining * 3, 256)
        points = sampler.random(n=batch_size)

        for point in points:
            config: Dict[str, int] = {}
            for dimension, key in enumerate(keys):
                choices = space[key]
                index = min(int(point[dimension] * len(choices)), len(choices) - 1)
                config[key] = int(choices[index])

            signature = tuple(config[key] for key in keys)
            if signature in seen:
                continue

            seen.add(signature)
            selected.append(config)
            if len(selected) == num_samples:
                break

    return selected


def select_workloads_from_shape_space(num_groups: int, seed: int) -> List[Workload]:
    """Select distinct workload triples from the built-in M/N/K shape grid."""
    all_workloads: List[Workload] = list(
        itertools.product(SHAPE_SPACE["M"], SHAPE_SPACE["N"], SHAPE_SPACE["K"])
    )
    if num_groups > len(all_workloads):
        raise ValueError(
            f"Requested {num_groups} groups, but only {len(all_workloads)} "
            "distinct workload triples are available."
        )

    rng = random.Random(seed)
    return rng.sample(all_workloads, num_groups)


def select_workloads_from_csv(
    csv_path: Path,
    num_groups: int,
    seed: int,
) -> List[Workload]:
    """Select distinct workload triples from an input CSV containing M/N/K."""
    workloads = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in GROUP_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Workloads CSV is missing required columns: {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                workload = tuple(int(float(row[column])) for column in GROUP_COLUMNS)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid M/N/K value on line {line_number} of {csv_path}"
                ) from None
            workloads.add(workload)

    candidates = sorted(workloads)
    if num_groups > len(candidates):
        raise ValueError(
            f"Requested {num_groups} groups, but {csv_path} contains only "
            f"{len(candidates)} unique (M, N, K) workloads."
        )

    rng = random.Random(seed)
    return rng.sample(candidates, num_groups)


def build_configurations(
    workloads: Iterable[Workload],
    configs_per_group: int,
    seed: int,
) -> List[Dict[str, int]]:
    """Create configs_per_group unique kernel configurations per workload."""
    rows: List[Dict[str, int]] = []

    for group_index, (m, n, k) in enumerate(workloads):
        # Separate deterministic seed per workload, so each group gets its own
        # reproducible LHS design.
        group_seed = seed + (group_index + 1) * 1009
        samples = discrete_lhs_samples(
            KERNEL_PARAM_SPACE,
            num_samples=configs_per_group,
            seed=group_seed,
        )

        for sample in samples:
            rows.append({"M": m, "N": n, "K": k, **sample})

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a GEMM CSV containing a fixed number of unique kernel "
            "configurations for each selected (M, N, K) workload."
        )
    )
    parser.add_argument("--groups", type=int, default=10, help="Number of distinct (M, N, K) workloads (default: 10).")
    parser.add_argument(
        "--configs-per-group",
        type=int,
        default=200,
        help="Number of unique kernel configurations per workload (default: 200).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gemm_grouped_based_configurations.csv"),
        help="Output CSV path (default: gemm_configurations.csv).",
    )
    parser.add_argument(
        "--workloads-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing M, N, K columns. If supplied, workload "
            "groups are selected from its unique triples instead of the built-in grid."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.groups < 1:
        print("Error: --groups must be at least 1.", file=sys.stderr)
        return 2
    if args.configs_per_group < 1:
        print("Error: --configs-per-group must be at least 1.", file=sys.stderr)
        return 2
    if args.workloads_csv is not None and not args.workloads_csv.is_file():
        print(f"Error: workloads CSV not found: {args.workloads_csv}", file=sys.stderr)
        return 2

    try:
        if args.workloads_csv is not None:
            workloads = select_workloads_from_csv(
                args.workloads_csv, args.groups, args.seed
            )
            source_description = str(args.workloads_csv)
        else:
            workloads = select_workloads_from_shape_space(args.groups, args.seed)
            source_description = "built-in GEMM shape space"

        rows = build_configurations(
            workloads,
            configs_per_group=args.configs_per_group,
            seed=args.seed,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    expected_rows = args.groups * args.configs_per_group
    if len(rows) != expected_rows:
        print(
            f"Internal error: expected {expected_rows} rows, generated {len(rows)}.",
            file=sys.stderr,
        )
        return 1

    print("GEMM configuration generation complete")
    print("-" * 64)
    print(f"Workload source                    : {source_description}")
    print(f"Number of distinct workload groups : {len(workloads)}")
    print(f"Configurations per workload       : {args.configs_per_group}")
    print(f"Total configurations              : {len(rows)}")
    print(f"Output file                        : {args.output}")
    print(f"Random seed                        : {args.seed}")
    print("-" * 64)
    print("Selected workloads (M, N, K):")
    for index, workload in enumerate(workloads, start=1):
        print(f"  {index:2d}. {workload}  -> {args.configs_per_group} configurations")

    print("\nAll generated parameter tuples are unique within each workload group.")
    print("Validity is not predicted here; your benchmark script will determine it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
