#!/usr/bin/env python3
"""Evaluate the GEMV XGBoost log-execution-time regression surrogate.

The script evaluates the saved LogExe model using the exact development/holdout
indices stored in the existing NPZ split file. It does not create a new split.

Run from the same directory as the final training script (default paths below):
    python evaluate_logexe_gemv.py
    python evaluate_logexe_gemv.py --threshold 12.5

Optional paths:
    python evaluate_logexe_gemv.py \
        --data ../datasets/t4/gemv/gemv_dataset.csv \
        --model ./xgb_logexe_model_gemv.json \
        --splits ./hyperparameter-tuning/gemv_group_cv_splits.npz

The text report is written to logexe_gemv_eval_report.txt by default.
"""

import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ndcg_score,
    r2_score,
    roc_auc_score,
)


# =============================================================================
# Configuration
# =============================================================================

INVALID_LAT = 1e8       # Penalty used by the training script for latency_ms == -1.0
VALID_CUTOFF = 1e7      # Preserve the original evaluation script's validity rule
LOG_INVALID = np.log(INVALID_LAT)

DEFAULT_DATA_PATH = "../../datasets/t4/gemv/gemv_dataset.csv"
DEFAULT_MODEL_PATH = "../xgb_logexe_model_gemv.json"
DEFAULT_SPLITS_PATH = "../hyperparameter-tuning/gemv_group_cv_splits.npz"

FEATURES = [
    "M", "K",
    "BLOCK_DIM_X", "ROWS_PER_BLOCK",
    "TILE_K", "UNROLL_FACTOR",
    "GRID_SIZE",
    "SHMEM_ELEMENTS",
    "WORK_PER_THREAD_K",
    "TILE_K_ALIGNED",
    "UNROLL_ALIGNMENT",
    "BOUNDARY_ALIGNMENT_M",
    "BOUNDARY_ALIGNMENT_K",
]

TOP_KS = (1, 5, 10)
NDCG_KS = (5, 10, 20)


# =============================================================================
# Report helpers
# =============================================================================

LINES = []


def out(message=""):
    print(message)
    LINES.append(str(message))


def header(title):
    out()
    out("=" * 82)
    out(title)
    out("=" * 82)


# =============================================================================
# Preprocessing -- keep this synchronized with the final training script
# =============================================================================


def preprocess_data_gemv(raw_df):
    keep_cols = [
        "M", "K",
        "BLOCK_DIM_X", "ROWS_PER_BLOCK",
        "TILE_K", "UNROLL_FACTOR",
        "latency_ms",
    ]

    df = raw_df[keep_cols].copy()

    # Apply the same invalid-configuration penalty used during training.
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = INVALID_LAT
    df["log_latency"] = np.log(df["latency_ms"])

    # Keep validity as an evaluation-only column (not an XGBoost feature).
    df["is_valid"] = df["latency_ms"] < VALID_CUTOFF

    # Derived features -- identical to the GEMV training script.
    df["GRID_SIZE"] = (
        (df["M"] + df["ROWS_PER_BLOCK"] - 1)
        // df["ROWS_PER_BLOCK"]
    )

    df["SHMEM_ELEMENTS"] = (
        df["TILE_K"] +
        (df["ROWS_PER_BLOCK"] * df["BLOCK_DIM_X"])
    )

    df["WORK_PER_THREAD_K"] = (
        (df["TILE_K"] + df["BLOCK_DIM_X"] - 1)
        // df["BLOCK_DIM_X"]
    )

    df["TILE_K_ALIGNED"] = (
        df["TILE_K"] % df["BLOCK_DIM_X"] == 0
    ).astype(int)

    df["UNROLL_ALIGNMENT"] = (
        df["WORK_PER_THREAD_K"] % df["UNROLL_FACTOR"] == 0
    ).astype(int)

    df["BOUNDARY_ALIGNMENT_M"] = (
        df["M"] % df["ROWS_PER_BLOCK"] == 0
    ).astype(int)

    df["BOUNDARY_ALIGNMENT_K"] = (
        df["K"] % df["TILE_K"] == 0
    ).astype(int)

    return df


# =============================================================================
# Metric helpers
# =============================================================================


def reg_metrics(true_log, pred_log):
    """Regression metrics in log space, with relative errors in latency space.

    MAPE/RPE are computed from expm1(pred_log - true_log). When invalid
    configurations are included, their target is the synthetic 1e8 penalty;
    combined MAPE should therefore be interpreted as a diagnostic, not as
    ordinary measured-latency percentage error.
    """
    true_log = np.asarray(true_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)

    if len(true_log) == 0:
        return None

    error = pred_log - true_log
    with np.errstate(over="ignore", invalid="ignore"):
        relative_error = np.abs(np.expm1(error))

    r2_log = float("nan")
    if len(true_log) >= 2 and np.ptp(true_log) > 0:
        r2_log = float(r2_score(true_log, pred_log))

    return {
        "n": len(true_log),
        "log_rmse": float(np.sqrt(np.mean(error ** 2))),
        "log_bias": float(np.mean(error)),
        "median_rpe": float(np.median(relative_error)),
        "mape": float(np.mean(relative_error)),
        "r2_log": r2_log,
    }


def show_reg(title, metrics):
    if metrics is None:
        out(f"{title}: no samples")
        return

    out(f"{title} (n = {metrics['n']})")
    out(f"  Log-space RMSE        : {metrics['log_rmse']:.4f}")
    out(
        f"  Mean log error (bias) : {metrics['log_bias']:+.4f}"
        "  (negative = model optimistic)"
    )
    out(f"  Median RPE            : {100 * metrics['median_rpe']:.2f} %")
    out(f"  MAPE                  : {100 * metrics['mape']:.2f} %")
    out(f"  R^2 (log space)       : {metrics['r2_log']:.4f}")


def show_train_test_rmse_gap(train_metrics, test_metrics, label):
    if train_metrics is None or test_metrics is None:
        out(f"Train/test log-RMSE gap ({label}): unavailable (no samples)")
        return

    gap = test_metrics["log_rmse"] - train_metrics["log_rmse"]
    out(
        f"Train/test log-RMSE gap ({label}; test - train): {gap:+.4f}"
    )


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def valid_values(values):
    """Return finite, non-None values for aggregation."""
    return np.asarray(
        [value for value in values
         if value is not None and np.isfinite(value)],
        dtype=float,
    )


def summ(values):
    values = valid_values(values)
    if len(values) == 0:
        return "n/a (no usable groups)"

    return (
        f"mean {values.mean():.4f} | median {np.median(values):.4f} "
        f"| std {values.std():.4f} | min {values.min():.4f} "
        f"| max {values.max():.4f} | groups {len(values)}"
    )


def pct_summ(values):
    values = valid_values(values)
    if len(values) == 0:
        return "n/a (no usable groups)"

    return (
        f"mean {values.mean():.4f} | median {np.median(values):.4f} "
        f"| 5th pct {np.percentile(values, 5):.4f} "
        f"| min {values.min():.4f} | groups {len(values)}"
    )


def relevance_labels(latencies):
    """Create per-workload NDCG relevance labels.

    Invalid configurations receive 0. Valid configurations receive grades
    from 1 (slowest) to 30 (fastest), using linear gain.
    """
    latencies = np.asarray(latencies, dtype=float)
    valid = latencies < VALID_CUTOFF
    relevance = np.zeros(len(latencies), dtype=float)

    n_valid = int(valid.sum())
    if n_valid >= 1:
        ranks = (
            pd.Series(latencies[valid]).rank(method="average").to_numpy() - 1.0
        )
        percentile_rank = ranks / max(n_valid - 1, 1)
        relevance[valid] = 1 + np.round(29 * (1 - percentile_rank))

    return relevance


def pairwise_accuracy(latencies, ranking_scores):
    """Fraction of unequal-latency pairs ordered correctly.

    Higher ranking_scores mean better configurations. Pairs with equal true
    latency are excluded. A predicted tie receives half credit. The loop
    uses O(n) temporary memory rather than building an O(n^2) matrix.
    """
    latencies = np.asarray(latencies, dtype=float)
    ranking_scores = np.asarray(ranking_scores, dtype=float)

    n = len(latencies)
    if n < 2:
        return np.nan

    correct = 0.0
    comparable_pairs = 0

    for i in range(n - 1):
        true_diff = latencies[i] - latencies[i + 1:]
        score_diff = ranking_scores[i] - ranking_scores[i + 1:]

        # Equal true latencies do not define a preferred ordering.
        not_true_tie = true_diff != 0
        if not np.any(not_true_tie):
            continue

        true_diff = true_diff[not_true_tie]
        score_diff = score_diff[not_true_tie]

        predicted_tie = score_diff == 0
        correctly_ordered = (
            ((true_diff < 0) & (score_diff > 0))
            | ((true_diff > 0) & (score_diff < 0))
        )

        correct += float(correctly_ordered.sum())
        correct += 0.5 * float(predicted_tie.sum())
        comparable_pairs += len(true_diff)

    if comparable_pairs == 0:
        return np.nan

    return correct / comparable_pairs


# =============================================================================
# Invalid-configuration classification helpers
# =============================================================================


def detection_row(true_invalid, predicted_score, threshold):
    predicted_invalid = predicted_score >= threshold

    tn, fp, fn, tp = confusion_matrix(
        true_invalid,
        predicted_invalid,
        labels=[False, True],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / len(true_invalid)

    return precision, recall, f1, accuracy, tp, fp, fn, tn


# =============================================================================
# Inference timing helper
# =============================================================================


def bench(fn, repeats, warmup=5):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    times = np.asarray(times, dtype=float)
    return float(np.median(times)), float(np.percentile(times, 95))


# =============================================================================
# Main evaluation
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the gemv XGBoost LogExe model on the saved holdout split."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--out", default="logexe_gemv_eval_report.txt")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Predicted log-latency threshold above which a configuration is "
            "classified as invalid. Default: midpoint between the slowest "
            "valid TRAIN config and log(1e8)."
        ),
    )
    parser.add_argument(
        "--good_frac",
        type=float,
        default=0.10,
        help="Fraction of fastest valid configurations per workload used as the good region.",
    )
    args = parser.parse_args()

    if not 0 < args.good_frac <= 1:
        parser.error("--good_frac must be greater than 0 and at most 1.")

    # -------------------------------------------------------------------------
    # Load data, preprocess, and use the EXISTING split from the NPZ file.
    # -------------------------------------------------------------------------
    raw_df = pd.read_csv(args.data)
    df = preprocess_data_gemv(raw_df)

    # Keep the same qid construction used by the training pipeline.
    df["qid"] = df.groupby(["M", "K"], sort=False).ngroup()

    with np.load(args.splits) as splits:
        development_indices = splits["development_indices"]
        holdout_indices = splits["holdout_indices"]

    train = df.iloc[development_indices].copy()
    test = df.iloc[holdout_indices].copy()

    # -------------------------------------------------------------------------
    # Load model and produce predictions for development and holdout rows.
    # -------------------------------------------------------------------------
    bst = xgb.Booster()
    bst.load_model(args.model)

    features = list(bst.feature_names) if bst.feature_names else FEATURES.copy()
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise SystemExit(
            f"Model expects features that are missing from the preprocessed data: "
            f"{missing_features}"
        )

    train["pred_log"] = bst.predict(xgb.DMatrix(train[features]))
    test["pred_log"] = bst.predict(xgb.DMatrix(test[features]))

    train_valid = train[train["is_valid"]].copy()
    test_valid = test[test["is_valid"]].copy()

    # -------------------------------------------------------------------------
    # Section 0: setup and sanity checks
    # -------------------------------------------------------------------------
    header("0. SETUP AND SANITY CHECKS")
    out(f"Dataset file        : {args.data}")
    out(f"Model file          : {args.model}")
    out(f"Split file          : {args.splits}")
    out(f"Boosted rounds      : {bst.num_boosted_rounds()}")
    out(f"XGBoost version     : {xgb.__version__}")
    out(f"Dataset rows        : {len(df)}")
    out(
        f"Train               : {len(train)} configs, "
        f"{train['qid'].nunique()} workloads"
    )
    out(
        f"Test                : {len(test)} configs, "
        f"{test['qid'].nunique()} workloads"
    )
    out(
        "Important: NPZ indices are positional. The CSV must have exactly the "
        "same rows in exactly the same order as when the split file was created."
    )

    train_keys = set(
        map(tuple, train[["M", "K"]].drop_duplicates().to_numpy())
    )
    test_keys = set(
        map(tuple, test[["M", "K"]].drop_duplicates().to_numpy())
    )
    shared_groups = len(train_keys & test_keys)
    out(f"Workloads shared between train and test: {shared_groups} (should be 0)")
    out(
        f"Invalid configurations: train {100 * (~train['is_valid']).mean():.2f} % "
        f"| test {100 * (~test['is_valid']).mean():.2f} %"
    )
    out(f"Valid test configurations: {len(test_valid)}")

    # -------------------------------------------------------------------------
    # Section 1: regression metrics on valid configurations only
    # -------------------------------------------------------------------------
    header("1. REGRESSION METRICS -- VALID CONFIGURATIONS ONLY")

    train_valid_metrics = reg_metrics(
        train_valid["log_latency"], train_valid["pred_log"]
    )
    test_valid_metrics = reg_metrics(
        test_valid["log_latency"], test_valid["pred_log"]
    )

    show_reg("TRAIN (development, in-sample)", train_valid_metrics)
    out()
    show_reg("TEST (held-out)", test_valid_metrics)
    out()
    show_train_test_rmse_gap(
        train_valid_metrics, test_valid_metrics, "valid configurations only"
    )
    out(
        "Note: Log RMSE, log bias and R^2 are calculated in log space. "
        "RPE/MAPE are relative errors after exponentiating the log-space error."
    )

    # -------------------------------------------------------------------------
    # Section 2: regression metrics on all configurations
    # -------------------------------------------------------------------------
    header("2. REGRESSION METRICS -- ALL CONFIGURATIONS (VALID + INVALID)")

    train_all_metrics = reg_metrics(train["log_latency"], train["pred_log"])
    test_all_metrics = reg_metrics(test["log_latency"], test["pred_log"])

    show_reg("TRAIN (development, in-sample)", train_all_metrics)
    out()
    show_reg("TEST (held-out)", test_all_metrics)
    out()
    show_train_test_rmse_gap(
        train_all_metrics, test_all_metrics, "valid + invalid configurations"
    )
    out(
        "Caution: invalid targets use the synthetic 1e8 latency penalty. "
        "All combined metrics therefore mix latency regression with the "
        "model's ability to reproduce the invalid-penalty target. Combined "
        "MAPE/RPE are relative errors against that penalty, not ordinary "
        "measured-latency percentage errors."
    )

    # -------------------------------------------------------------------------
    # Section 3: accuracy in the good region
    # -------------------------------------------------------------------------
    header(
        f"3. ACCURACY IN THE GOOD REGION "
        f"(FASTEST {100 * args.good_frac:.0f}% VALID CONFIGS PER MATRIX SIZE)"
    )

    if len(test_valid) == 0:
        out("No valid held-out configurations; good-region metrics are unavailable.")
    else:
        good_mask = np.zeros(len(test_valid), dtype=bool)
        test_valid_for_good = test_valid.reset_index(drop=True)

        for _, indices in test_valid_for_good.groupby("qid", sort=False).indices.items():
            n_good = max(1, int(np.ceil(args.good_frac * len(indices))))
            order = indices[
                np.argsort(
                    test_valid_for_good["latency_ms"].to_numpy()[indices],
                    kind="stable",
                )
            ]
            good_mask[order[:n_good]] = True

        good_region = test_valid_for_good[good_mask]
        show_reg(
            "TEST, good region",
            reg_metrics(good_region["log_latency"], good_region["pred_log"]),
        )
        out(
            "Note: R^2 can be low or negative here because the true latency range "
            "is deliberately narrow. RMSE, RPE, MAPE and bias are more useful."
        )

    # -------------------------------------------------------------------------
    # Section 4: error spread across held-out workloads
    # -------------------------------------------------------------------------
    header("4. ERROR SPREAD ACROSS HELD-OUT WORKLOADS (VALID CONFIGS ONLY)")

    per_group = []
    for qid, group in test_valid.groupby("qid", sort=False):
        error = group["pred_log"].to_numpy() - group["log_latency"].to_numpy()
        group_rmse = float(np.sqrt(np.mean(error ** 2)))
        per_group.append(
            (
                qid,
                group["M"].iloc[0],
                group["K"].iloc[0],
                group_rmse,
                len(group),
            )
        )

    if per_group:
        group_rmse_values = np.asarray([item[3] for item in per_group], dtype=float)
        out(
            f"Per-workload log-RMSE: mean {group_rmse_values.mean():.4f} "
            f"| median {np.median(group_rmse_values):.4f} "
            f"| std {group_rmse_values.std():.4f} "
            f"| 5th pct {np.percentile(group_rmse_values, 5):.4f} "
            f"| 95th pct {np.percentile(group_rmse_values, 95):.4f} "
            f"| min {group_rmse_values.min():.4f} "
            f"| max {group_rmse_values.max():.4f} "
            f"| workloads {len(per_group)}"
        )
        out("Worst 5 workloads (M, K, log-RMSE, #valid configs):")
        for _, m, k, rmse, n_configs in sorted(
            per_group, key=lambda item: -item[3]
        )[:5]:
            out(
                f"M={int(m)}, K={int(k)} : "
                f"{rmse:.4f} (n={n_configs})"
            )
    else:
        out("No valid held-out workloads available for per-workload error analysis.")

    # -------------------------------------------------------------------------
    # Section 5: invalid-configuration detection
    # -------------------------------------------------------------------------
    header("5. INVALID-CONFIGURATION DETECTION (THRESHOLD ON PREDICTED LOG-LATENCY)")

    if len(train_valid) == 0:
        out("No valid training configurations; a training-derived threshold cannot be set.")
    else:
        valid_train_max = float(train_valid["log_latency"].max())
        threshold = (
            args.threshold
            if args.threshold is not None
            else (valid_train_max + LOG_INVALID) / 2
        )

        out(f"Slowest valid TRAIN config (log): {valid_train_max:.3f}")
        out(f"Invalid penalty (log)           : {LOG_INVALID:.3f}")
        out(
            f"Threshold used                  : {threshold:.3f} "
            f"({'user-given' if args.threshold is not None else 'training-derived midpoint'})"
        )

        true_invalid = (~test["is_valid"]).to_numpy(dtype=bool)
        invalid_score = test["pred_log"].to_numpy(dtype=float)

        if true_invalid.sum() == 0 or true_invalid.all():
            out(
                "Test set has only one class (all valid or all invalid); "
                "ROC-AUC, Average Precision and threshold metrics are skipped."
            )
        else:
            out(f"ROC-AUC (threshold-free)       : {roc_auc_score(true_invalid, invalid_score):.4f}")
            out(
                f"Average Precision (PR-AUC)     : "
                f"{average_precision_score(true_invalid, invalid_score):.4f}"
            )
            out()

            precision, recall, f1, accuracy, tp, fp, fn, tn = detection_row(
                true_invalid, invalid_score, threshold
            )
            out(f"At the chosen threshold ({threshold:.3f}):")
            out(f"  Precision : {precision:.4f}")
            out(f"  Recall    : {recall:.4f} (fraction of invalid configs caught)")
            out(f"  F1        : {f1:.4f}")
            out(f"  Accuracy  : {accuracy:.4f}")
            out(f"  Confusion : TP {tp} | FP {fp} | FN {fn} | TN {tn}")
            out(
                f"  FN = invalid configs predicted valid: {fn} of {tp + fn} "
                f"({100 * fn / max(tp + fn, 1):.2f} %)"
            )
            out()
            out("Threshold sweep (sanity check):")
            out(f"  {'threshold':>10} {'precision':>10} {'recall':>10} {'F1':>10}")

            sweep_start = valid_train_max
            sweep_end = LOG_INVALID - 0.5
            if sweep_start < sweep_end:
                for sweep_threshold in np.linspace(sweep_start, sweep_end, 6):
                    p, r, f, _, _, _, _, _ = detection_row(
                        true_invalid, invalid_score, float(sweep_threshold)
                    )
                    out(
                        f"  {sweep_threshold:10.2f} {p:10.4f} "
                        f"{r:10.4f} {f:10.4f}"
                    )
            else:
                out("  Threshold sweep skipped: sweep start is not below sweep end.")

    # -------------------------------------------------------------------------
    # Section 6: ranking metrics on LogExe predictions
    # -------------------------------------------------------------------------
    header("6. RANKING METRICS ON LOG-EXE PREDICTIONS (GROUPED BY MATRIX SIZE)")
    out(
        "Ranking direction: lower predicted log-latency is better. "
        "For shared ranking metrics, ranking_score = -pred_log."
    )

    spearman_all = []
    spearman_valid = []
    pairwise_all = []
    pairwise_valid = []

    ndcg_all = {k: [] for k in list(NDCG_KS) + [None]}
    ndcg_valid = {k: [] for k in list(NDCG_KS) + [None]}

    fraction_optimal = {k: [] for k in TOP_KS}
    invalid_candidate_fraction = {k: [] for k in TOP_KS}
    groups_with_invalid_in_topk = {k: [] for k in TOP_KS}

    for _, group in test.groupby("qid", sort=False):
        predicted_log = group["pred_log"].to_numpy(dtype=float)
        ranking_scores = -predicted_log
        latencies = group["latency_ms"].to_numpy(dtype=float)
        valid = group["is_valid"].to_numpy(dtype=bool)

        # Grouped Spearman: positive correlation means predicted and true
        # latency increase together, so lower predictions align with lower truth.
        spearman_all.append(safe_spearman(predicted_log, latencies))
        if valid.sum() >= 3:
            spearman_valid.append(
                safe_spearman(predicted_log[valid], latencies[valid])
            )

        # Pairwise accuracy uses a higher-is-better ranking score.
        pairwise_all.append(pairwise_accuracy(latencies, ranking_scores))
        if valid.sum() >= 2:
            pairwise_valid.append(
                pairwise_accuracy(latencies[valid], ranking_scores[valid])
            )

        # NDCG: shared relevance labels for all models in future comparisons.
        relevance = relevance_labels(latencies)
        if len(group) >= 2 and valid.any():
            for k in list(NDCG_KS) + [None]:
                ndcg_all[k].append(
                    ndcg_score(relevance[None, :], ranking_scores[None, :], k=k)
                )

        if valid.sum() >= 2:
            valid_relevance = relevance[valid]
            valid_scores = ranking_scores[valid]
            for k in list(NDCG_KS) + [None]:
                ndcg_valid[k].append(
                    ndcg_score(
                        valid_relevance[None, :], valid_scores[None, :], k=k
                    )
                )

        # Top-k selection metrics are computed over all configurations. A
        # selected invalid configuration remains in the top-k denominator.
        order = np.argsort(predicted_log, kind="stable")
        for k in TOP_KS:
            selected_count = min(k, len(group))
            selected = order[:selected_count]
            selected_valid = valid[selected]

            invalid_fraction = 1.0 - float(selected_valid.mean())
            invalid_candidate_fraction[k].append(invalid_fraction)
            groups_with_invalid_in_topk[k].append(
                float(np.any(~selected_valid))
            )

            # Fraction of optimal is only defined for workloads with at least
            # one valid configuration in the ground truth.
            if valid.any():
                true_optimum = float(latencies[valid].min())
                best_selected_latency = float(latencies[selected].min())
                fraction_optimal[k].append(
                    true_optimum / best_selected_latency
                )

    out("Spearman correlation (predicted latency vs true latency; higher is better):")
    out(f"  Grouped, all configs (invalid included): {summ(spearman_all)}")
    out(f"  Grouped, valid configs only           : {summ(spearman_valid)}")

    pooled_all_spearman = safe_spearman(
        test["pred_log"].to_numpy(), test["latency_ms"].to_numpy()
    )
    pooled_valid_spearman = safe_spearman(
        test_valid["pred_log"].to_numpy(), test_valid["latency_ms"].to_numpy()
    )
    out(f"  Pooled global, all configs (supplementary): {pooled_all_spearman:.4f}")
    out(f"  Pooled global, valid only (supplementary)  : {pooled_valid_spearman:.4f}")
    out(
        "  Pooled values mix workloads and can reflect the fact that larger "
        "matrices are slower; grouped values are the primary results."
    )
    out()

    out("Pairwise accuracy (unequal-latency pairs only; predicted ties receive 0.5):")
    out(f"  Grouped, all configs (invalid included): {summ(pairwise_all)}")
    out(f"  Grouped, valid configs only           : {summ(pairwise_valid)}")
    out()

    out(
        "NDCG (invalid = 0; valid = 1..30 based on within-workload latency rank; "
        "linear gain):"
    )
    for label, store in (
        ("all configs (invalid included)", ndcg_all),
        ("valid configs only", ndcg_valid),
    ):
        out(f"  {label}:")
        for k in list(NDCG_KS) + [None]:
            metric_name = f"NDCG@{k}" if k is not None else "NDCG (full)"
            out(f"    {metric_name:<12}: {summ(store[k])}")
    out()

    out("Top-k fraction of optimal (1.0 = true optimum selected):")
    out("  Formula: true workload optimum / best true latency among predicted top-k.")
    for k in TOP_KS:
        out(f"  Top-{k:<2}: {pct_summ(fraction_optimal[k])}")
    out()

    out("Invalid selections in the model's predicted top-k configurations:")
    out("  Candidate fraction = mean invalid fraction among the selected top-k.")
    out("  Workload rate      = fraction of workloads with at least one invalid in top-k.")
    for k in TOP_KS:
        out(
            f"  Top-{k:<2} invalid candidate fraction: "
            f"{100 * np.mean(invalid_candidate_fraction[k]):.2f} % "
            f"({summ(invalid_candidate_fraction[k])})"
            if invalid_candidate_fraction[k]
            else f"  Top-{k:<2} invalid candidate fraction: n/a"
        )
        out(
            f"       workloads with >=1 invalid: "
            f"{100 * np.mean(groups_with_invalid_in_topk[k]):.2f} % "
            f"({summ(groups_with_invalid_in_topk[k])})"
            if groups_with_invalid_in_topk[k]
            else "       workloads with >=1 invalid: n/a"
        )

    # -------------------------------------------------------------------------
    # Section 7: inference latency
    # -------------------------------------------------------------------------
    header("7. INFERENCE LATENCY (XGBOOST PREDICTION)")
    out(
        f"xgboost {xgb.__version__} | python {platform.python_version()} "
        f"| CPU logical cores: {os.cpu_count()} "
        f"| {platform.processor() or platform.machine()}"
    )
    out(
        "Timing measures XGBoost inference and, separately, DMatrix construction "
        "plus inference. It does not include the full PSO candidate-generation pipeline."
    )

    X = test[features].to_numpy(dtype=np.float32)
    check_count = min(256, len(X))
    dm_check = xgb.DMatrix(X[:check_count], feature_names=list(features))
    numpy_predictions = bst.predict(dm_check)
    dataframe_predictions = test["pred_log"].to_numpy()[:check_count]
    max_difference = (
        float(np.max(np.abs(numpy_predictions - dataframe_predictions)))
        if check_count
        else 0.0
    )
    out(
        "Sanity: NumPy-DMatrix vs DataFrame-DMatrix predictions, "
        f"max absolute difference = {max_difference:.2e}"
    )
    out()

    if len(X) == 0:
        out("No holdout configurations available; inference benchmark skipped.")
    else:
        batch_sizes = list(dict.fromkeys([1, 32, 128, 1024, len(X)]))
        out(
            f"{'batch':>7} | "
            f"{'predict only: median / p95 ms':^31} | "
            f"{'DMatrix + predict: median / p95 ms':^37}"
        )
        out(
            f"{'':>7} | {'ms/call':>9} {'p95 ms':>9} {'us/config':>10} "
            f"| {'ms/call':>9} {'p95 ms':>9} {'us/config':>10}"
        )

        for batch_size in batch_sizes:
            indices = np.resize(np.arange(len(X)), batch_size)
            batch_X = X[indices]
            batch_dmatrix = xgb.DMatrix(
                batch_X,
                feature_names=list(features),
            )
            repeats = (
                300 if batch_size == 1
                else 100 if batch_size <= 128
                else 20
            )

            predict_median, predict_p95 = bench(
                lambda: bst.predict(batch_dmatrix), repeats
            )
            full_median, full_p95 = bench(
                lambda: bst.predict(
                    xgb.DMatrix(batch_X, feature_names=list(features))
                ),
                repeats,
            )

            out(
                f"{batch_size:7d} | "
                f"{1e3 * predict_median:9.3f} {1e3 * predict_p95:9.3f} "
                f"{1e6 * predict_median / batch_size:10.2f} | "
                f"{1e3 * full_median:9.3f} {1e3 * full_p95:9.3f} "
                f"{1e6 * full_median / batch_size:10.2f}"
            )

        out(
            "Median and 95th-percentile times are measured after warm-up. "
            "The DMatrix + predict column includes input-matrix construction."
        )

    # -------------------------------------------------------------------------
    # Write text report
    # -------------------------------------------------------------------------
    output_directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(LINES) + "\n")

    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
