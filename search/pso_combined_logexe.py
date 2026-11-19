"""Evaluate combined LogExe + PSO across GEMM, BMM and GEMV on a T4."""
import argparse
import importlib.util
import json
from pathlib import Path
from shutil import which

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PARTICLE_COUNTS = [10, 30, 50, 100]
N_RUNS = 25
N_ITERS = 50


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def combined_feature_builder(tuning, kernel, workload_columns):
    def build(configs, shape, features):
        raw = pd.DataFrame(configs)
        for column, value in zip(workload_columns, shape):
            raw[column] = value
        # Preprocessing also constructs training targets; a dummy positive label
        # is discarded before inference and cannot enter the model's inputs.
        raw["latency_ms"] = 1.0
        return tuning.preprocess(raw, kernel).loc[:, features]
    return build


def workload_membership(dev, holdout, kernel, columns):
    membership = {}
    for label, frame in (("development", dev), ("holdout", holdout)):
        for shape in frame.loc[frame.kernel == kernel, columns].drop_duplicates().itertuples(index=False, name=None):
            membership[tuple(map(int, shape))] = label
    return membership


def write_outputs(details, summaries, output, settings, cache_count):
    runs = pd.concat(details, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    runs.to_csv(output / "pso_combined_run_details.csv", index=False)
    summary.to_csv(output / "pso_combined_summary.csv", index=False)
    # Reuse each kernel's summary definitions with a single artificial workload
    # to aggregate all equally repeated workloads at each particle count.
    aggregate = []
    for kernel, group in runs.groupby("kernel", sort=False):
        helper = settings[kernel]
        pooled = group.copy()
        for column in helper.WORKLOAD_COLUMNS:
            pooled[column] = 0
        table = helper.make_summary(pooled)
        table = table.drop(columns=helper.WORKLOAD_COLUMNS + ["baseline_ms", "best_seen_latency_ms"])
        table.insert(0, "kernel", kernel)
        aggregate.append(table)
    aggregate = pd.concat(aggregate, ignore_index=True)
    aggregate.to_csv(output / "pso_combined_kernel_summary.csv", index=False)
    with (output / "pso_combined_report.txt").open("w", encoding="utf-8") as report:
        report.write("Combined LogExe PSO evaluation\n")
        report.write(f"Runs: {len(runs)} | Particles: {PARTICLE_COUNTS} | Repeats: {N_RUNS} | Iterations: {N_ITERS}\n")
        report.write(f"Distinct recommendations benchmarked: {cache_count}\n")
        report.write("Best seen is the best valid measurement in each workload's reference CSV.\n")
        report.write("Negative gap means the recommendation beats the sampled reference.\n")
        report.write("Near-optimal rates count all attempts; invalid rates use returned configurations.\n")
        report.write("Latency/speedup averages cover successful valid benchmarks.\n")
        report.write("Tuning time includes PSO and feature construction; predict time is a subset.\n")
        report.write("Model loading and subsequent CUDA compilation/measurement are excluded.\n")
        report.write("Identical kernel/workload/configuration recommendations reuse a GPU measurement.\n")
        report.write("Kernel summaries pool runs with equal repeats per workload.\n")
        for title, table in (("PER-KERNEL SUMMARY", aggregate), ("PER-WORKLOAD SUMMARY", summary)):
            report.write(f"\n{title}\n")
            for row in table.to_dict("records"):
                report.write("\n")
                for key, value in row.items():
                    if pd.notna(value):
                        report.write(f"  {key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/xgb_logexe_model_combined.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--kernel", choices=["gemm", "bmm", "gemv"],
                           help="Run one kernel independently.")
    selection.add_argument("--kernels", nargs="+", choices=["gemm", "bmm", "gemv"],
                           help="Run a selected list of kernels (default: all three).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder; defaults to a folder for the selected kernels.")
    args = parser.parse_args()
    args.kernels = [args.kernel] if args.kernel else (args.kernels or ["gemm", "bmm", "gemv"])
    if len(set(args.kernels)) != len(args.kernels):
        parser.error("--kernels must not contain duplicates")
    if args.out is None:
        args.out = ROOT / "search/pso_evaluations/combined" / "_".join(args.kernels)
    training = load_module(ROOT / "models/xgb_logexe_train_combined.py", "combined_training")
    tuning, manifest, _, dev, holdout = training.load_training_inputs(args.kernels)
    features = manifest["features"]
    model = xgb.XGBRegressor()
    model.load_model(str(args.model))
    if model.get_booster().feature_names != features:
        raise ValueError("Combined model features/order differ from the tuning manifest.")

    jobs, helpers = [], {}
    for kernel in args.kernels:
        helper = load_module(ROOT / f"search/pso_{kernel}_logexe.py", f"combined_{kernel}_helpers")
        helper.N_ITERS = N_ITERS
        helper.validate_search_space()
        data_path = ROOT / f"datasets/t4/{kernel}/group_based/{kernel}_group_based_dataset.csv"
        baseline_path = ROOT / f"datasets/t4/{kernel}/group_based/{kernel}_baseline_results.csv"
        # Workload count comes from the actual reference CSV.
        helper.EXPECTED_DATASET_ROWS = len(pd.read_csv(data_path))
        data = helper.load_dataset(data_path)
        shapes = sorted(data[helper.WORKLOAD_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
        baseline = helper.load_baseline_times(baseline_path)
        helper.validate_baselines(shapes, baseline)
        best = helper.get_best_seen_by_workload(data)
        if any(shape not in best for shape in shapes):
            raise ValueError(f"{kernel}: a reference workload has no valid measurements.")
        source = ROOT / f"kernels/{kernel}/main.cu"
        if not source.is_file() or which(helper.NVCC) is None:
            raise FileNotFoundError(f"{kernel}: CUDA source and nvcc are required.")
        # This isolated helper module keeps its original search, seeds, timers and
        # benchmark logic, while its fitness uses the combined feature builder.
        helper.build_feature_frame = combined_feature_builder(tuning, kernel, helper.WORKLOAD_COLUMNS)
        membership = workload_membership(dev, holdout, kernel, helper.WORKLOAD_COLUMNS)
        helpers[kernel] = helper
        jobs.append((kernel, helper, shapes, baseline, best, source, membership))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pso_combined_settings.json").write_text(json.dumps({
        "model": str(args.model), "features": features, "kernels": args.kernels,
        "particles": PARTICLE_COUNTS, "runs": N_RUNS, "iterations": N_ITERS,
        "search": {k: {"space": h.SEARCH_SPACE, "seed": h.BASE_RANDOM_SEED,
                       "c1": h.PSO_C1, "c2": h.PSO_C2, "w": h.PSO_W,
                       "k": h.PSO_K, "p": h.PSO_P, "cuda_arch": h.CUDA_ARCH}
                   for k, h in helpers.items()},
    }, indent=2), encoding="utf-8")
    details, summaries, cache = [], [], {}
    total = sum(len(job[2]) for job in jobs) * N_RUNS * len(PARTICLE_COUNTS)
    completed = 0
    print(f"Running {total} combined-model PSO trials.", flush=True)
    for kernel, helper, shapes, baseline, best, source, membership in jobs:
        rows = []
        for workload_index, shape in enumerate(shapes):
            for particles in PARTICLE_COUNTS:
                for run in range(1, N_RUNS + 1):
                    result = helper.run_single_pso(model, shape, features, particles, run, workload_index)
                    config = result.pop("config")
                    row = dict(zip(helper.WORKLOAD_COLUMNS, shape))
                    row.update(result)
                    row.update(kernel=kernel, membership=membership.get(shape, "unseen"),
                               n_particles=particles, run=run, pso_iterations=N_ITERS,
                               benchmark_status="pso_failed", benchmark_error="",
                               benchmark_cache_hit=False, actual_latency_ms=np.nan)
                    if config is not None:
                        row.update(config)
                    rows.append(row)
                    completed += 1
                    if completed % 25 == 0:
                        print(f"{kernel}: PSO {completed}/{total}", flush=True)
        # GPU verification follows the search runs, as in the separate evaluators.
        for index, row in enumerate(rows):
            if row["pso_status"] != "success":
                continue
            shape = tuple(int(row[c]) for c in helper.WORKLOAD_COLUMNS)
            names = helper.TUNABLE_PARAMETERS + ([] if kernel == "gemv" else ["THREAD_Z"])
            config = {name: int(row[name]) for name in names}
            key = (kernel, *helper.config_cache_key(shape, config))
            row["benchmark_cache_hit"] = key in cache
            if key not in cache:
                cache[key] = helper.benchmark_final_config(shape, config, source)
                print(f"{kernel}: GPU measurement {len(cache)} completed ({index + 1}/{len(rows)} runs)", flush=True)
            row.update(cache[key])
        frame = helper.add_run_metrics(pd.DataFrame(rows), best, baseline)
        summary = helper.make_summary(frame)
        summary.insert(0, "kernel", kernel)
        summary["membership"] = [
            membership.get(tuple(int(row[c]) for c in helper.WORKLOAD_COLUMNS), "unseen")
            for _, row in summary.iterrows()
        ]
        summary["best_seen_config_json"] = [
            json.dumps(best[tuple(int(row[c]) for c in helper.WORKLOAD_COLUMNS)]["best_seen_config"])
            for _, row in summary.iterrows()
        ]
        details.append(frame)
        summaries.append(summary)
        write_outputs(details, summaries, args.out, helpers, len(cache))
    print(f"Reports saved to {args.out}")


if __name__ == "__main__":
    main()
