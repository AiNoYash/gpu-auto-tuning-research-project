#!/usr/bin/env python3
"""Benchmark one fixed baseline BMM configuration on every workload in a CSV.

The CSV must contain M, N, K, and BATCH_SIZE columns. Each unique (M, N, K, BATCH_SIZE) tuple is
benchmarked once. The CUDA executable performs its own warm-up and repeated
kernel timing, and this script records the median_ms it prints.
"""

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_CSV = SCRIPT_DIR / "../../datasets/t4/bmm/group_based/bmm_group_based_dataset.csv"
CUDA_SOURCE = SCRIPT_DIR / "main.cu"
OUTPUT_CSV = SCRIPT_DIR / Path("bmm_baseline_results.csv")
EXECUTABLE = SCRIPT_DIR / Path("bmm_baseline")

NVCC = "nvcc"
CUDA_ARCH = "sm_75"  # NVIDIA Tesla T4
RUN_TIMEOUT_SECONDS = 300

# Baseline configuration: these match the fallback defaults in the supplied
# main.cu source. They are a reasonable, valid starting reference for T4,
# not a claim that this is the fastest configuration.
BASELINE_CONFIG = {
    "TILE_M": 64,
    "TILE_N": 64,
    "TILE_K": 16,
    "THREAD_X": 16,
    "THREAD_Y": 16,
    "THREAD_Z": 1,
    "UNROLL_FACTOR": 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_unique_workloads(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path.resolve()}")

    workloads = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"M", "N", "K", "BATCH_SIZE"}
        available = set(reader.fieldnames or [])
        missing = required - available
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                # Handles both "1024" and "1024.0" CSV representations.
                shape = tuple(int(float(row[name])) for name in ("M", "N", "K", "BATCH_SIZE"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid M/N/K/BATCH_SIZE value at CSV row {row_number}: {row}"
                ) from exc

            if any(dimension <= 0 for dimension in shape):
                raise ValueError(
                    f"M, N, K, and BATCH_SIZE must be positive at CSV row {row_number}: {shape}"
                )
            workloads.add(shape)

    if not workloads:
        raise ValueError("No workloads were found in the dataset CSV.")

    return sorted(workloads)


def validate_baseline_config():
    threads_per_block = (
        BASELINE_CONFIG["THREAD_X"]
        * BASELINE_CONFIG["THREAD_Y"]
        * BASELINE_CONFIG["THREAD_Z"]
    )
    shared_memory_bytes = (
        BASELINE_CONFIG["TILE_M"] * BASELINE_CONFIG["TILE_K"]
        + BASELINE_CONFIG["TILE_K"] * BASELINE_CONFIG["TILE_N"]
    ) * 4  # float32

    if threads_per_block > 1024:
        raise ValueError(
            f"Baseline uses {threads_per_block} threads per block; maximum is 1024."
        )
    if shared_memory_bytes > 48 * 1024:
        raise ValueError(
            "Baseline requests more than 48 KiB of shared memory: "
            f"{shared_memory_bytes} bytes."
        )


def compile_baseline(cuda_source: Path, executable: Path):
    if not cuda_source.is_file():
        raise FileNotFoundError(f"CUDA source not found: {cuda_source.resolve()}")

    nvcc_path = shutil.which(NVCC)
    if nvcc_path is None:
        raise FileNotFoundError(
            f"Could not find '{NVCC}' on PATH. Activate your CUDA environment first."
        )

    command = [
        nvcc_path,
        "-O3",
        str(cuda_source),
        f"-arch={CUDA_ARCH}",
        "-o",
        str(executable),
    ]
    for parameter, value in BASELINE_CONFIG.items():
        command.append(f"-D{parameter}={value}")

    print("Compiling the baseline CUDA executable...")
    print("Command:", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            "Baseline CUDA compilation failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if completed.stderr.strip():
        print("Compiler messages:")
        print(completed.stderr.strip())


def parse_cuda_json(stdout: str):
    """Parse the JSON object printed by main.cu, allowing preceding log lines."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not find a valid JSON object in stdout: {stdout!r}")


def benchmark_workload(executable: Path, shape):
    m, n, k, batch_size = shape
    command = [str(executable.resolve()), str(m), str(n), str(k), str(batch_size)]
    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
        wall_time_s = time.perf_counter() - start

        if completed.returncode != 0:
            return {
                "latency_ms": -1.0,
                "status": "process_failed",
                "mean_ms": -1.0,
                "variance": -1.0,
                "rel_variance": -1.0,
                "iterations": 0,
                "measurement_wall_s": wall_time_s,
                "error_message": completed.stderr.strip()
                or f"Executable returned code {completed.returncode}",
            }

        try:
            payload = parse_cuda_json(completed.stdout)
        except ValueError as exc:
            return {
                "latency_ms": -1.0,
                "status": "json_parse_failed",
                "mean_ms": -1.0,
                "variance": -1.0,
                "rel_variance": -1.0,
                "iterations": 0,
                "measurement_wall_s": wall_time_s,
                "error_message": str(exc),
            }

        status = payload.get("status", "unknown")
        if status == "success":
            return {
                "latency_ms": float(payload["median_ms"]),
                "status": status,
                "mean_ms": float(payload.get("mean_ms", -1.0)),
                "variance": float(payload.get("variance", -1.0)),
                "rel_variance": float(payload.get("rel_variance", -1.0)),
                "iterations": int(payload.get("iterations", 0)),
                "measurement_wall_s": wall_time_s,
                "error_message": "",
            }

        return {
            "latency_ms": -1.0,
            "status": status,
            "mean_ms": -1.0,
            "variance": -1.0,
            "rel_variance": -1.0,
            "iterations": 0,
            "measurement_wall_s": wall_time_s,
            "error_message": payload.get("error_message", "Unknown CUDA error"),
        }

    except subprocess.TimeoutExpired:
        return {
            "latency_ms": -1.0,
            "status": "timeout",
            "mean_ms": -1.0,
            "variance": -1.0,
            "rel_variance": -1.0,
            "iterations": 0,
            "measurement_wall_s": RUN_TIMEOUT_SECONDS,
            "error_message": f"Execution exceeded {RUN_TIMEOUT_SECONDS} seconds.",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    validate_baseline_config()
    workloads = read_unique_workloads(DATASET_CSV)

    print(f"Found {len(workloads)} unique BMM workloads.")
    if len(workloads) != 10:
        print(
            f"WARNING: Expected 10 workloads based on your dataset description; "
            f"found {len(workloads)}. All discovered workloads will be benchmarked."
        )

    print("Baseline configuration:", BASELINE_CONFIG)
    compile_baseline(CUDA_SOURCE, EXECUTABLE)

    output_fields = [
        "M", "N", "K", "BATCH_SIZE",
        *BASELINE_CONFIG.keys(),
        "latency_ms", "status", "mean_ms", "variance", "rel_variance",
        "iterations", "measurement_wall_s", "error_message",
    ]

    results = []
    for index, shape in enumerate(workloads, start=1):
        m, n, k, batch_size = shape
        print(f"\n[{index}/{len(workloads)}] Benchmarking M={m}, N={n}, K={k}, BATCH_SIZE={batch_size} ...")
        result = benchmark_workload(EXECUTABLE, shape)
        row = {"M": m, "N": n, "K": k, "BATCH_SIZE": batch_size, **BASELINE_CONFIG, **result}
        results.append(row)

        if result["status"] == "success":
            print(f"  Baseline latency: {result['latency_ms']:.6f} ms")
            print(f"  CUDA timing iterations: {result['iterations']}")
        else:
            print(
                f"  Benchmark failed: {result['status']} - "
                f"{result['error_message']}"
            )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved baseline results to: {OUTPUT_CSV.resolve()}")
    failed = sum(row["status"] != "success" for row in results)
    print(f"\nCompleted: {len(results) - failed}/{len(results)} successful baselines.")
    if failed:
        print("Resolve failed baseline measurements before using them in speedup calculations.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
