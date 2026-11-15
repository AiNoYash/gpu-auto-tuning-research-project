#!/usr/bin/env python3
"""Summarize configuration counts grouped by GPU-kernel workload.

Usage (with this script placed alongside the existing evaluation scripts):
    python analyze_kernel_dataset.py gemm
    python analyze_kernel_dataset.py bmm
    python analyze_kernel_dataset.py gemv

Optional arguments:
    python analyze_kernel_dataset.py gemm --data /path/to/gemm_dataset.csv
    python analyze_kernel_dataset.py bmm --data-root /path/to/datasets/t4
    python analyze_kernel_dataset.py gemv --invalid-value -1.0

By default, the CSV path is constructed relative to the current working directory:
    ../../datasets/t4/<kernel>/<kernel>_dataset.csv

The project convention is that latency_ms == -1.0 marks an invalid
configuration. Other finite, non-negative latency values are treated as valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Workload-defining keys for each kernel type.
GROUP_COLUMNS = {
    "gemm": ["M", "N", "K"],
    "bmm": ["BATCH_SIZE", "M", "N", "K"],
    "gemv": ["M", "K"],
}

LATENCY_COLUMN = "latency_ms"
DEFAULT_INVALID_VALUE = -1.0

# Matches the relative dataset paths used by the existing evaluation scripts.
# This path is resolved relative to the directory from which the script is run.
DEFAULT_DATA_ROOT = Path("../datasets/t4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze valid/invalid configurations and workload-group sizes "
            "for a GEMM, BMM, or GEMV dataset."
        )
    )
    parser.add_argument(
        "kernel",
        choices=sorted(GROUP_COLUMNS),
        help="Kernel dataset to analyze: gemm, bmm, or gemv.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Explicit CSV path. Overrides the automatically constructed path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root directory containing gemm/, bmm/, and gemv/ folders. "
            f"Default (relative to current working directory): {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--invalid-value",
        type=float,
        default=DEFAULT_INVALID_VALUE,
        help=f"Latency sentinel for invalid configurations (default: {DEFAULT_INVALID_VALUE}).",
    )
    return parser.parse_args()


def resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data is not None:
        return args.data.expanduser().resolve()

    data_root = args.data_root.expanduser().resolve()
    return data_root / args.kernel / f"{args.kernel}_dataset.csv"


def analyze_dataset(
    df: pd.DataFrame,
    kernel: str,
    invalid_value: float,
) -> dict[str, object]:
    group_columns = GROUP_COLUMNS[kernel]

    required_columns = group_columns + [LATENCY_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required column(s): " + ", ".join(missing_columns)
        )

    if df.empty:
        return {
            "group_columns": group_columns,
            "total_configs": 0,
            "total_groups": 0,
            "invalid_configs": 0,
            "valid_configs": 0,
            "unclassified_configs": 0,
            "max_configs_per_group": 0,
            "invalid_only_groups": 0,
            "single_config_groups": 0,
        }

    # Missing workload keys cannot be assigned reliably to a workload group.
    rows_with_missing_keys = int(df[group_columns].isna().any(axis=1).sum())
    if rows_with_missing_keys:
        raise ValueError(
            f"{rows_with_missing_keys} row(s) have missing workload key(s) "
            f"in {group_columns}. Fix or remove these rows before analysis."
        )

    # Convert latency to numeric. Non-numeric and missing values are reported
    # separately as unclassified rather than silently counted as valid/invalid.
    latency = pd.to_numeric(df[LATENCY_COLUMN], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(latency)

    is_invalid = finite & (latency == invalid_value)
    is_valid = finite & (latency >= 0.0) & (latency != invalid_value)
    is_unclassified = ~(is_invalid | is_valid)

    analysis_df = df[group_columns].copy()
    analysis_df["_is_invalid"] = is_invalid
    analysis_df["_is_valid"] = is_valid
    analysis_df["_is_unclassified"] = is_unclassified

    # Each unique workload-key combination defines one group.
    per_group = (
        analysis_df.groupby(group_columns, sort=False, dropna=False)
        .agg(
            config_count=("_is_invalid", "size"),
            invalid_count=("_is_invalid", "sum"),
            valid_count=("_is_valid", "sum"),
            unclassified_count=("_is_unclassified", "sum"),
        )
    )

    group_sizes = per_group["config_count"]
    invalid_only = per_group["invalid_count"] == per_group["config_count"]

    return {
        "group_columns": group_columns,
        "total_configs": int(len(df)),
        "total_groups": int(len(per_group)),
        "invalid_configs": int(is_invalid.sum()),
        "valid_configs": int(is_valid.sum()),
        "unclassified_configs": int(is_unclassified.sum()),
        "max_configs_per_group": int(group_sizes.max()) if len(group_sizes) else 0,
        "invalid_only_groups": int(invalid_only.sum()),
        "single_config_groups": int((group_sizes == 1).sum()),
    }


def print_report(kernel: str, data_path: Path, invalid_value: float, stats: dict[str, object]) -> None:
    print("=" * 68)
    print(f"{kernel.upper()} DATASET GROUP ANALYSIS")
    print("=" * 68)
    print(f"Dataset file                       : {data_path}")
    print(f"Grouping keys                     : {', '.join(stats['group_columns'])}")
    print(f"Invalid latency sentinel           : {invalid_value}")
    print("-" * 68)
    print(f"Total configurations              : {stats['total_configs']:,}")
    print(f"Total groups                      : {stats['total_groups']:,}")
    print(f"Valid configurations              : {stats['valid_configs']:,}")
    print(f"Invalid configurations            : {stats['invalid_configs']:,}")
    print(f"Unclassified configurations       : {stats['unclassified_configs']:,}")
    print("-" * 68)
    print(f"Maximum configs in a single group : {stats['max_configs_per_group']:,}")
    print(f"Groups containing only invalids   : {stats['invalid_only_groups']:,}")
    print(f"Groups containing exactly 1 config: {stats['single_config_groups']:,}")
    print("=" * 68)

    if stats["unclassified_configs"]:
        print(
            "Note: Unclassified rows have missing/non-numeric latency values or "
            "finite negative latency values other than the invalid sentinel."
        )


def main() -> int:
    args = parse_args()
    data_path = resolve_data_path(args)

    if not data_path.is_file():
        print(f"Error: dataset file not found: {data_path}", file=sys.stderr)
        print(
            "Check --data / --data-root, or run it from the project directory where "
            "../../datasets/t4 points to the dataset directory.",
            file=sys.stderr,
        )
        return 1

    try:
        df = pd.read_csv(data_path)
        stats = analyze_dataset(df, args.kernel, args.invalid_value)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(args.kernel, data_path, args.invalid_value, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
