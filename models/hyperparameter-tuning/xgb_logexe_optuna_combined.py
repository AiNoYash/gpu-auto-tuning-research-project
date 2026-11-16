"""Tune one LogExe model for GEMM, BMM and GEMV using their saved grouped splits."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "combined_logexe"
KERNELS = ("gemm", "bmm", "gemv")
WORKLOADS = {
    "gemm": ["M", "N", "K"],
    "bmm": ["BATCH_SIZE", "M", "N", "K"],
    "gemv": ["M", "K"],
}
DATASETS = {k: ROOT / f"datasets/t4/{k}/{k}_dataset.csv" for k in KERNELS}
SPLITS = {k: Path(__file__).resolve().parent / f"{k}_group_cv_splits.npz" for k in KERNELS}
BASELINES = {
    k: ROOT / f"datasets/t4/{k}/group_based/{k}_baseline_results.csv"
    for k in KERNELS
}
N_TRIALS = 300
N_SPLITS = 5
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 75
FINAL_ESTIMATOR_MULTIPLIER = 1.1
SEED = 42
INVALID_LATENCY = 1e8
FEATURES = [
    "is_gemm", "is_bmm", "is_gemv", "BATCH_SIZE", "M", "N", "K",
    "TILE_M", "TILE_N", "TILE_K", "THREAD_X", "THREAD_Y", "THREAD_Z",
    "BLOCK_DIM_X", "ROWS_PER_BLOCK", "UNROLL_FACTOR", "TOTAL_THREADS",
    "GRID_SIZE", "SHMEM_BYTES", "WORK_PER_THREAD_M", "WORK_PER_THREAD_N",
    "WORK_PER_THREAD_K", "TILE_K_ALIGNED", "UNROLL_ALIGNMENT",
    "BOUNDARY_ALIGNMENT_M", "BOUNDARY_ALIGNMENT_N", "BOUNDARY_ALIGNMENT_K",
]


def preprocess(raw, kernel):
    """Preserve source row order and construct shared and kernel-specific features."""
    df = raw.copy()
    df["kernel"] = kernel
    df["source_row"] = np.arange(len(df))
    df["is_valid"] = df["latency_ms"] != -1
    if not np.isfinite(df["latency_ms"]).all() or (
        df.loc[df["is_valid"], "latency_ms"] <= 0
    ).any():
        raise ValueError(f"{kernel}: latency must be positive or -1.")
    df["log_latency"] = np.log(df["latency_ms"].where(df["is_valid"], INVALID_LATENCY))
    for name in KERNELS:
        df[f"is_{name}"] = int(kernel == name)
    if kernel != "bmm":
        df["BATCH_SIZE"] = 1
    if kernel == "gemv":
        df["N"] = 1
    # Initialize inapplicable inputs explicitly, even if the CSV has extra columns.
    inactive = (
        ["TILE_M", "TILE_N", "THREAD_X", "THREAD_Y", "THREAD_Z",
         "WORK_PER_THREAD_M", "WORK_PER_THREAD_N", "BOUNDARY_ALIGNMENT_N"]
        if kernel == "gemv" else
        ["BLOCK_DIM_X", "ROWS_PER_BLOCK", "WORK_PER_THREAD_K",
         "TILE_K_ALIGNED", "BOUNDARY_ALIGNMENT_K"]
    )
    for name in inactive:
        df[name] = np.nan
    if kernel == "gemv":
        block, rows, tile = df["BLOCK_DIM_X"], df["ROWS_PER_BLOCK"], df["TILE_K"]
        df["TOTAL_THREADS"] = block
        df["GRID_SIZE"] = (df["M"] + rows - 1) // rows
        df["SHMEM_BYTES"] = 4 * (tile + rows * block)
        df["WORK_PER_THREAD_K"] = (tile + block - 1) // block
        df["TILE_K_ALIGNED"] = (tile % block == 0).astype(int)
        df["UNROLL_ALIGNMENT"] = (df["WORK_PER_THREAD_K"] % df["UNROLL_FACTOR"] == 0).astype(int)
        df["BOUNDARY_ALIGNMENT_M"] = (df["M"] % rows == 0).astype(int)
        df["BOUNDARY_ALIGNMENT_K"] = (df["K"] % tile == 0).astype(int)
    else:
        df["TOTAL_THREADS"] = df["THREAD_X"] * df["THREAD_Y"] * df["THREAD_Z"]
        df["GRID_SIZE"] = np.ceil(df["M"] / df["TILE_M"]) * np.ceil(df["N"] / df["TILE_N"]) * df["BATCH_SIZE"]
        df["SHMEM_BYTES"] = 4 * (df["TILE_M"] * df["TILE_K"] + df["TILE_K"] * df["TILE_N"])
        df["WORK_PER_THREAD_M"] = (df["TILE_M"] + df["THREAD_Y"] - 1) // df["THREAD_Y"]
        df["WORK_PER_THREAD_N"] = (df["TILE_N"] + df["THREAD_X"] - 1) // df["THREAD_X"]
        df["UNROLL_ALIGNMENT"] = (df["TILE_K"] % df["UNROLL_FACTOR"] == 0).astype(int)
        df["BOUNDARY_ALIGNMENT_M"] = (df["M"] % df["TILE_M"] == 0).astype(int)
        df["BOUNDARY_ALIGNMENT_N"] = (df["N"] % df["TILE_N"] == 0).astype(int)
    return df


def load_data():
    development, holdout, fold_parts, provenance = [], [], [[] for _ in range(N_SPLITS)], {}
    offset = 0
    for kernel in KERNELS:
        raw = pd.read_csv(DATASETS[kernel])
        df = preprocess(raw, kernel)
        with np.load(SPLITS[kernel]) as saved:
            dev, test = saved["development_indices"], saved["holdout_indices"]
            if not np.array_equal(np.sort(np.concatenate([dev, test])), np.arange(len(df))):
                raise ValueError(f"{kernel}: saved split does not partition this dataset.")
            groups = df.groupby(WORKLOADS[kernel], sort=False).ngroup().to_numpy()
            if set(groups[dev]) & set(groups[test]):
                raise ValueError(f"{kernel}: workload leakage in saved outer split.")
            val_rows = []
            for fold in range(N_SPLITS):
                train = saved[f"fold_{fold + 1}_train_idx"]
                val = saved[f"fold_{fold + 1}_val_idx"]
                if not np.array_equal(np.sort(np.concatenate([train, val])), np.arange(len(dev))):
                    raise ValueError(f"{kernel}: invalid saved CV fold.")
                if set(groups[dev[train]]) & set(groups[dev[val]]):
                    raise ValueError(f"{kernel}: workload leakage in saved CV fold.")
                fold_parts[fold].append((train + offset, val + offset))
                val_rows.extend(val.tolist())
            if sorted(val_rows) != list(range(len(dev))):
                raise ValueError(f"{kernel}: validation folds do not partition development rows.")
        development.append(df.iloc[dev])
        holdout.append(df.iloc[test])
        offset += len(dev)
        provenance[kernel] = {
            "dataset": str(DATASETS[kernel]), "split": str(SPLITS[kernel]),
            "dataset_sha256": hashlib.sha256(DATASETS[kernel].read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(SPLITS[kernel].read_bytes()).hexdigest(),
            "rows": len(df), "development_rows": len(dev), "holdout_rows": len(test),
        }
    folds = [(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
             for parts in fold_parts]
    return pd.concat(development, ignore_index=True), pd.concat(holdout, ignore_index=True), folds, provenance


def load_baselines():
    result = {}
    for kernel, path in BASELINES.items():
        if not path.exists():
            result[kernel] = {}
            continue
        df = pd.read_csv(path)
        df = df.loc[(df["status"] == "success") & np.isfinite(df["latency_ms"]) & (df["latency_ms"] > 0)]
        if df.duplicated(WORKLOADS[kernel]).any():
            raise ValueError(f"Duplicate baseline workloads in {path}")
        result[kernel] = {
            tuple(int(row[c]) for c in WORKLOADS[kernel]): float(row["latency_ms"])
            for _, row in df.iterrows()
        }
    return result


def selection_metrics(validation, predictions, baselines):
    """Macro-average selection outcomes across workloads within each kernel."""
    df = validation.copy()
    df["prediction"] = predictions
    result = {}
    for kernel, data in df.groupby("kernel"):
        valid = data["is_valid"]
        errors = data.loc[valid, "prediction"] - data.loc[valid, "log_latency"]
        counts = dict(workloads=0, evaluable_workloads=0, invalid=0, near5=0, near10=0)
        regrets, speedups = [], []
        for shape, group in data.groupby(WORKLOADS[kernel], sort=False):
            counts["workloads"] += 1
            winner = group.iloc[np.argmin(group["prediction"].to_numpy())]
            counts["invalid"] += int(not winner["is_valid"])
            measured = group.loc[group["is_valid"], "latency_ms"]
            if len(measured):
                counts["evaluable_workloads"] += 1
                if winner["is_valid"]:
                    regret = float(winner["latency_ms"] / measured.min() - 1)
                    regrets.append(regret)
                    counts["near5"] += int(regret <= .05)
                    counts["near10"] += int(regret <= .10)
            baseline = baselines[kernel].get(tuple(shape))
            if baseline is not None and winner["is_valid"]:
                speedups.append(float(baseline / winner["latency_ms"]))
        n = counts["evaluable_workloads"]
        result[kernel] = {
            **counts,
            "valid_log_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "mean_top1_regret_valid_selections": float(np.mean(regrets)) if regrets else None,
            "valid_regret_workloads": len(regrets),
            "within_5pct_rate": counts["near5"] / n if n else None,
            "within_10pct_rate": counts["near10"] / n if n else None,
            "invalid_top1_rate": counts["invalid"] / counts["workloads"],
            "mean_speedup_vs_default_valid_selections": float(np.mean(speedups)) if speedups else None,
            "speedup_workloads": len(speedups),
            "baseline_covered_workloads": sum(tuple(s) in baselines[kernel] for s in data.groupby(WORKLOADS[kernel]).groups),
        }
    return result


def main():
    import optuna
    import xgboost as xgb

    dev, holdout, folds, provenance = load_data()
    baselines = load_baselines()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_output = {}
    for i, (train, val) in enumerate(folds, 1):
        split_output[f"fold_{i}_train_idx"] = train
        split_output[f"fold_{i}_val_idx"] = val
    np.savez(OUTPUT_DIR / "combined_cv_splits.npz", **split_output)
    for label, frame in (("development", dev), ("holdout", holdout)):
        frame[["kernel", "source_row"]].to_csv(OUTPUT_DIR / f"{label}_row_mapping.csv", index=False)
    manifest = {
        "features": FEATURES, "sources": provenance, "seed": SEED,
        "max_boost_rounds": MAX_BOOST_ROUNDS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "final_estimator_multiplier": FINAL_ESTIMATOR_MULTIPLIER,
        "preprocessing_function": "preprocess in " + str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "missing": "NaN for inapplicable parameters and derived features",
        "defaults": {"non_bmm_batch_size": 1, "gemv_N": 1},
        "shared_memory_element_bytes": 4, "invalid_latency": INVALID_LATENCY,
        "target": "natural log latency_ms after invalid penalty substitution",
        "objective": "mean of five pooled validation log RMSE values including invalid penalties",
        "selection_metrics": "Per-kernel validation workloads; measured candidates only; ties use source row order. Invalid selections fail near-optimal rates. Regret and speedup means cover valid selections only.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    matrices = [(xgb.DMatrix(dev.iloc[t][FEATURES], label=dev.iloc[t]["log_latency"]),
                 xgb.DMatrix(dev.iloc[v][FEATURES], label=dev.iloc[v]["log_latency"]))
                for t, v in folds]

    def objective(trial):
        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
            "seed": SEED, "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", .01, .15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", .6, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .6, 1),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        }
        scores, rounds, caps, diagnostics = [], [], [], []
        for i, ((train, val), (_, val_indices)) in enumerate(zip(matrices, folds)):
            model = xgb.train(params, train, num_boost_round=MAX_BOOST_ROUNDS,
                              evals=[(val, "validation")],
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)
            best_rounds = model.best_iteration + 1
            prediction = model.predict(val, iteration_range=(0, best_rounds))
            scores.append(float(model.best_score))
            rounds.append(best_rounds)
            caps.append(model.num_boosted_rounds() == MAX_BOOST_ROUNDS)
            diagnostics.append(selection_metrics(dev.iloc[val_indices], prediction, baselines))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        for key, value in dict(fold_rmses=scores, fold_best_rounds=rounds,
                               fold_reached_cap=caps, fold_selection_metrics=diagnostics).items():
            trial.set_user_attr(key, value)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    final_rounds = int(np.ceil(np.median(best.user_attrs["fold_best_rounds"]) * FINAL_ESTIMATOR_MULTIPLIER))
    params = dict(best.params, objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", seed=SEED, n_estimators=final_rounds)
    summary = dict(best_cv_rmse=best.value, best_trial=best.number, **best.user_attrs,
                   final_n_estimators=final_rounds)
    (OUTPUT_DIR / "best_params.json").write_text(json.dumps(params, indent=2))
    (OUTPUT_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    study.trials_dataframe().to_csv(OUTPUT_DIR / "trials.csv", index=False)
    print(json.dumps(summary, indent=2))
    if any(best.user_attrs["fold_reached_cap"]):
        print("Some winning-trial folds reached the boosting cap. Inspect their best rounds before increasing it.")
    print(f"Saved combined tuning outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
