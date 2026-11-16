#!/usr/bin/env python3
"""Measure one fixed GEMV baseline configuration for every workload in a CSV.

The input CSV must contain M and K columns. Each unique (M, K) workload is
benchmarked once. The GEMV CUDA executable is compiled once with the fixed
baseline parameters, then run for each workload. The program's median_ms is
recorded as latency_ms.

The baseline values below are a reasonable starting reference for an NVIDIA
T4, not a claim of optimality. Verify them against the defaults/constraints in
your GEMV .cu source and your actual parameter space.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Read the configurations CSV you use for GEMV dataset generation.
# It only needs to contain M and K; the other columns are ignored here.
DATASET_CSV = Path("../../datasets/t4/gemv/group_based/gemv_group_based_dataset.csv")

# Set this to the GEMV CUDA source file (not the GEMM source file).
CUDA_SOURCE = Path("./main.cu")

OUTPUT_CSV = Path("gemv_baseline_results.csv")
EXECUTABLE = Path("gemv_baseline")

NVCC = "nvcc"
CUDA_ARCH = "sm_75"  # NVIDIA Tesla T4
COMPILE_TIMEOUT_SECONDS = 120.0
RUN_TIMEOUT_SECONDS = 100.0

# Fixed GEMV baseline. These use a standard 128-thread block, four rows per
# block, a 256-element K tile, and an unroll factor of two. They are reference
# settings only; benchmark them rather than assuming they are optimal.
BASELINE_CONFIG: dict[str, int] = {
    "BLOCK_DIM_X": 128,
    "ROWS_PER_BLOCK": 4,
    "TILE_K": 256,
    "UNROLL_FACTOR": 2,
}

EXPECTED_WORKLOADS = 10  # Set to None to disable the count warning.


# =============================================================================
# HELPERS
# =============================================================================


def resolve_path(path: Path) -> Path:
    """Resolve relative paths from the directory containing this script."""
    return path if path.is_absolute() else (Path(__file__).resolve().parent / path).resolve()


def parse_int(value: Any, column: str, row_number: int) -> int:
    """Accept integer CSV values represented as either '128' or '128.0'."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {column} value at CSV row {row_number}: {value!r}"
        ) from exc

    if not number.is_integer():
        raise ValueError(
            f"Expected integer {column} at CSV row {row_number}, got {value!r}"
        )
    return int(number)


def read_unique_workloads(csv_path: Path) -> list[tuple[int, int]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset/configuration CSV not found: {csv_path}")

    workloads: set[tuple[int, int]] = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        available = set(reader.fieldnames or [])
        missing = {"M", "K"} - available
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            m = parse_int(row.get("M"), "M", row_number)
            k = parse_int(row.get("K"), "K", row_number)
            if m <= 0 or k <= 0:
                raise ValueError(
                    f"M and K must be positive at CSV row {row_number}: {(m, k)}"
                )
            workloads.add((m, k))

    if not workloads:
        raise ValueError("No GEMV workloads were found in the input CSV.")

    return sorted(workloads)


def validate_baseline_config() -> None:
    required = {"BLOCK_DIM_X", "ROWS_PER_BLOCK", "TILE_K", "UNROLL_FACTOR"}
    if set(BASELINE_CONFIG) != required:
        raise ValueError(
            f"BASELINE_CONFIG must contain exactly these keys: {sorted(required)}"
        )

    for name, value in BASELINE_CONFIG.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")

    if BASELINE_CONFIG["BLOCK_DIM_X"] > 1024:
        raise ValueError("BLOCK_DIM_X exceeds the CUDA maximum of 1024 threads/block.")

    if BASELINE_CONFIG["BLOCK_DIM_X"] % 32 != 0:
        warnings.warn(
            "BLOCK_DIM_X is not a multiple of the NVIDIA warp size (32). "
            "This may be intentional, but verify that it is suitable for your kernel.",
            RuntimeWarning,
        )


def compile_baseline(cuda_source: Path, executable: Path) -> None:
    if not cuda_source.is_file():
        raise FileNotFoundError(
            f"GEMV CUDA source not found: {cuda_source}. Set CUDA_SOURCE correctly."
        )

    nvcc_path = shutil.which(NVCC)
    if nvcc_path is None:
        raise FileNotFoundError(
            f"Could not find {NVCC!r} on PATH. Activate your CUDA environment first."
        )

    command = [
        nvcc_path,
        "-O3",
        str(cuda_source),
        f"-arch={CUDA_ARCH}",
        "-o",
        str(executable),
    ]
    command.extend(
        f"-D{name}={value}" for name, value in BASELINE_CONFIG.items()
    )

    print("\nCompiling GEMV baseline executable...")
    print("Command:", " ".join(command))

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"NVCC compilation exceeded {COMPILE_TIMEOUT_SECONDS} seconds."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "GEMV baseline compilation failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if completed.stderr.strip():
        print("Compiler messages:")
        print(completed.stderr.strip())

    print("Compilation successful.")


def parse_cuda_json(stdout: str) -> dict[str, Any]:
    """Find and parse the JSON object printed by the CUDA program."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue

    # Also handle output where the JSON object spans multiple lines.
    first = stdout.find("{")
    last = stdout.rfind("}")
    if first >= 0 and last > first:
        payload = json.loads(stdout[first:last + 1])
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"Could not find a valid JSON object in stdout: {stdout!r}")


def benchmark_workload(
    executable: Path, workload: tuple[int, int]
) -> dict[str, Any]:
    m, k = workload
    command = [str(executable.resolve()), str(m), str(k)]
    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        measurement_wall_s = time.perf_counter() - start

        if completed.returncode != 0:
            return failure_result(
                "process_failed",
                measurement_wall_s,
                completed.stderr.strip()
                or f"CUDA executable returned code {completed.returncode}.",
            )

        try:
            payload = parse_cuda_json(completed.stdout)
        except (ValueError, json.JSONDecodeError) as exc:
            return failure_result("json_parse_failed", measurement_wall_s, str(exc))

        status = str(payload.get("status", "unknown_status"))
        if status != "success":
            return failure_result(
                status,
                measurement_wall_s,
                str(payload.get("error_message", "Unknown CUDA error.")),
            )

        try:
            latency_ms = float(payload["median_ms"])
            mean_ms = float(payload.get("mean_ms", -1.0))
            variance = float(payload.get("variance", -1.0))
            rel_variance = float(payload.get("rel_variance", -1.0))
            iterations = int(payload.get("iterations", 0))
        except (KeyError, TypeError, ValueError) as exc:
            return failure_result(
                "invalid_json_metrics", measurement_wall_s,
                f"Missing or invalid timing fields: {exc}",
            )

        if latency_ms <= 0:
            return failure_result(
                "invalid_latency_value", measurement_wall_s,
                f"CUDA returned median_ms={latency_ms!r}.",
            )

        return {
            "latency_ms": latency_ms,
            "status": status,
            "mean_ms": mean_ms,
            "variance": variance,
            "rel_variance": rel_variance,
            "iterations": iterations,
            "measurement_wall_s": measurement_wall_s,
            "error_message": "",
        }

    except subprocess.TimeoutExpired:
        return failure_result(
            "timeout",
            RUN_TIMEOUT_SECONDS,
            f"GEMV executable exceeded {RUN_TIMEOUT_SECONDS} seconds.",
        )


def failure_result(status: str, wall_time_s: float, message: str) -> dict[str, Any]:
    return {
        "latency_ms": -1.0,
        "status": status,
        "mean_ms": -1.0,
        "variance": -1.0,
        "rel_variance": -1.0,
        "iterations": 0,
        "measurement_wall_s": wall_time_s,
        "error_message": message,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    validate_baseline_config()

    dataset_csv = resolve_path(DATASET_CSV)
    cuda_source = resolve_path(CUDA_SOURCE)
    output_csv = resolve_path(OUTPUT_CSV)
    executable = resolve_path(EXECUTABLE)

    workloads = read_unique_workloads(dataset_csv)
    print(f"Found {len(workloads)} unique GEMV workloads (M, K).")

    if EXPECTED_WORKLOADS is not None and len(workloads) != EXPECTED_WORKLOADS:
        print(
            f"WARNING: Expected {EXPECTED_WORKLOADS} workloads, found {len(workloads)}. "
            "All discovered workloads will be benchmarked."
        )

    print("Baseline configuration:", BASELINE_CONFIG)
    compile_baseline(cuda_source, executable)

    output_fields = [
        "M", "K",
        *BASELINE_CONFIG.keys(),
        "latency_ms", "status", "mean_ms", "variance", "rel_variance",
        "iterations", "measurement_wall_s", "error_message",
    ]

    results: list[dict[str, Any]] = []
    for index, workload in enumerate(workloads, start=1):
        m, k = workload
        print(f"\n[{index}/{len(workloads)}] Benchmarking GEMV workload M={m}, K={k} ...")
        measured = benchmark_workload(executable, workload)

        row: dict[str, Any] = {
            "M": m,
            "K": k,
            **BASELINE_CONFIG,
            **measured,
        }
        results.append(row)

        if measured["status"] == "success":
            print(
                f"  Success: {measured['latency_ms']:.6f} ms | "
                f"runs={measured['iterations']} | "
                f"relative_variance={measured['rel_variance']:.6g}"
            )
        else:
            print(
                f"  Failed: status={measured['status']} | "
                f"{measured['error_message']}"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 72)
    print(f"Saved baseline results to: {output_csv}")
    print("\nBASELINE_TIMES_MS dictionary (copy into your GEMV PSO evaluation script):")
    print("BASELINE_TIMES_MS = {")
    for row in results:
        if row["status"] == "success":
            print(
                f"    ({row['M']}, {row['K']}): "
                f"{row['latency_ms']:.10g},"
            )
        else:
            print(
                f"    # ({row['M']}, {row['K']}): measurement failed "
                f"({row['status']}); rerun/check this workload."
            )
    print("}")
    print("=" * 72)

    # Keep the executable, matching the GEMM baseline script's behavior.
    print(f"\nCompiled executable: {executable}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
