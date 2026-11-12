#!/usr/bin/env python3
"""Evaluate the GEMM XGBoost pairwise-ranking surrogate.

The script evaluates the saved rank:pairwise model using the exact
 development/holdout indices stored in the existing NPZ split file. It does not
create a new split.

Run from the same directory as the final GEMM training scripts:
    python evaluate_rank_gemm.py

Optional paths:
    python evaluate_rank_gemm.py \
        --data ../datasets/t4/gemm/gemm_dataset.csv \
        --model ./xgb_rank_model_gemm.json \
        --splits ./hyperparameter-tuning/gemm_group_cv_splits.npz

The text report is written to rank_gemm_eval_report.txt by default.
"""

import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


# =============================================================================
# Configuration
# =============================================================================

INVALID_LAT = 1e8
VALID_CUTOFF = 1e7

DEFAULT_DATA_PATH = "../../datasets/t4/gemm/gemm_dataset.csv"
DEFAULT_MODEL_PATH = "../xgb_rank_model_gemm.json"
DEFAULT_SPLITS_PATH = "../hyperparameter-tuning/gemm_group_cv_splits.npz"

# Keep this feature list synchronized with the GEMM rank training script.
FEATURES = [
    "M", "N", "K",
    "TILE_M", "TILE_N", "TILE_K",
    "THREAD_X", "THREAD_Y", "THREAD_Z",
    "UNROLL_FACTOR",
    "TOTAL_THREADS", "GRID_SIZE",
    "WORK_PER_THREAD_M", "WORK_PER_THREAD_N",
    "SHMEM_FOOTPRINT", "UNROLL_ALIGNMENT",
    "BOUNDARY_ALIGNMENT_M", "BOUNDARY_ALIGNMENT_N",
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
    out("=" * 86)
    out(title)
    out("=" * 86)


def valid_values(values):
    """Return finite values for aggregation."""
    return np.asarray(
        [value for value in values
         if value is not None and np.isfinite(value)],
        dtype=float,
    )


def summ(values):
    """Summarize per-workload values, with each workload weighted equally."""
    values = valid_values(values)
    if len(values) == 0:
        return "n/a (no usable workloads)"

    return (
        f"mean {values.mean():.4f} | median {np.median(values):.4f} "
        f"| std {values.std():.4f} | min {values.min():.4f} "
        f"| max {values.max():.4f} | workloads {len(values)}"
    )


def pct_summ(values):
    """Summarize values expressed as fractions, and display them as percent."""
    values = valid_values(values)
    if len(values) == 0:
        return "n/a (no usable workloads)"

    values = 100.0 * values
    return (
        f"mean {values.mean():.2f}% | median {np.median(values):.2f}% "
        f"| std {values.std():.2f} pp | min {values.min():.2f}% "
        f"| max {values.max():.2f}% | workloads {len(values)}"
    )


# =============================================================================
# Preprocessing -- identical to the GEMM training pipeline
# =============================================================================


def preprocess_data_gemm(raw_df):
    keep_cols = [
        "M", "N", "K",
        "TILE_M", "TILE_N", "TILE_K",
        "THREAD_X", "THREAD_Y", "THREAD_Z",
        "UNROLL_FACTOR",
        "latency_ms",
    ]

    df = raw_df[keep_cols].copy()

    # Apply the same invalid-configuration penalty used during training.
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = INVALID_LAT

    # Evaluation-only validity flag; it is NOT an XGBoost feature.
    df["is_valid"] = df["latency_ms"] < VALID_CUTOFF

    # Derived features -- same formulas as the GEMM training pipeline.
    df["TOTAL_THREADS"] = (
        df["THREAD_X"] * df["THREAD_Y"] * df["THREAD_Z"]
    )

    df["GRID_SIZE"] = (
        np.ceil(df["M"] / df["TILE_M"])
        * np.ceil(df["N"] / df["TILE_N"])
    )

    df["WORK_PER_THREAD_M"] = (
        (df["TILE_M"] + df["THREAD_Y"] - 1) // df["THREAD_Y"]
    )

    df["WORK_PER_THREAD_N"] = (
        (df["TILE_N"] + df["THREAD_X"] - 1) // df["THREAD_X"]
    )

    df["SHMEM_FOOTPRINT"] = (
        (df["TILE_M"] * df["TILE_K"])
        + (df["TILE_K"] * df["TILE_N"])
    )

    df["UNROLL_ALIGNMENT"] = (
        (df["TILE_K"] % df["UNROLL_FACTOR"] == 0).astype(int)
    )

    df["BOUNDARY_ALIGNMENT_M"] = (
        (df["M"] % df["TILE_M"] == 0).astype(int)
    )

    df["BOUNDARY_ALIGNMENT_N"] = (
        (df["N"] % df["TILE_N"] == 0).astype(int)
    )

    return df


# =============================================================================
# Ranking metric helpers
# =============================================================================


def safe_spearman(a, b):
    """Spearman correlation, returning NaN when it is not defined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan

    return float(spearmanr(a, b)[0])


def relevance_labels(latencies):
    """Create shared NDCG labels from measured latency within one workload.

    Invalid configurations receive relevance 0. Valid configurations receive
    grades 1..30, with 30 assigned to the fastest valid configuration. Linear
    gain is used (matching ndcg_exp_gain=False in XGBoost training).
    """
    latencies = np.asarray(latencies, dtype=float)
    valid = latencies < VALID_CUTOFF
    relevance = np.zeros(len(latencies), dtype=float)

    n_valid = int(valid.sum())
    if n_valid:
        ranks = (
            pd.Series(latencies[valid]).rank(method="average").to_numpy() - 1.0
        )
        percentile_rank = ranks / max(n_valid - 1, 1)
        relevance[valid] = 1 + np.round(29 * (1 - percentile_rank))

    return relevance


def pairwise_accuracy(latencies, ranking_scores):
    """Accuracy over pairs with unequal true latency.

    Higher ranking scores mean better configurations. A predicted tie receives
    half credit. Temporary memory is O(n), although runtime is O(n^2).
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


def calculate_workload_metrics(group, good_frac):
    """Calculate ranking and top-k selection metrics for one workload."""
    scores = group["ranking_score"].to_numpy(dtype=float)
    latencies = group["latency_ms"].to_numpy(dtype=float)
    valid = group["is_valid"].to_numpy(dtype=bool)

    # Higher score means better. -latency is also higher for faster configs.
    metrics = {
        "spearman_all": safe_spearman(scores, -latencies),
        "spearman_valid": safe_spearman(scores[valid], -latencies[valid]),
        "pairwise_all": pairwise_accuracy(latencies, scores),
        "pairwise_valid": pairwise_accuracy(latencies[valid], scores[valid]),
        "ndcg_all": np.nan,
        "ndcg_valid": np.nan,
    }

    relevance = relevance_labels(latencies)
    if len(group) >= 2 and valid.any():
        metrics["ndcg_all"] = float(
            ndcg_score(relevance[None, :], scores[None, :])
        )

    if valid.sum() >= 2:
        metrics["ndcg_valid"] = float(
            ndcg_score(relevance[valid][None, :], scores[valid][None, :])
        )

    # Predicted ordering: rank:pairwise outputs higher scores for better configs.
    predicted_order = np.argsort(-scores, kind="stable")

    # Define the good region as the fastest good_frac of VALID configurations.
    good_mask = np.zeros(len(group), dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        n_good = max(1, int(np.ceil(good_frac * len(valid_indices))))
        true_valid_order = valid_indices[
            np.argsort(latencies[valid_indices], kind="stable")
        ]
        good_indices = true_valid_order[:n_good]
        good_mask[good_indices] = True
    else:
        n_good = 0

    metrics["ndcg_at_10"] = np.nan
    if len(group) >= 2 and valid.any():
        metrics["ndcg_at_10"] = float(
            ndcg_score(relevance[None, :], scores[None, :], k=10)
        )

    for k in TOP_KS:
        selected_count = min(k, len(group))
        selected = predicted_order[:selected_count]
        selected_valid = valid[selected]

        metrics[f"invalid_fraction_{k}"] = (
            1.0 - float(selected_valid.mean()) if selected_count else np.nan
        )
        metrics[f"has_invalid_{k}"] = (
            float(np.any(~selected_valid)) if selected_count else np.nan
        )

        if valid.any():
            true_optimum = float(latencies[valid].min())
            best_selected_latency = float(latencies[selected].min())

            # Hit = at least one configuration with the exact best measured
            # latency appears in the model's top-k. isclose handles float CSVs.
            metrics[f"hit_{k}"] = float(
                np.any(
                    np.isclose(
                        latencies[selected],
                        true_optimum,
                        rtol=1e-9,
                        atol=1e-12,
                    )
                )
            )
            metrics[f"regret_{k}"] = (
                best_selected_latency / true_optimum
            ) - 1.0
            metrics[f"fraction_optimal_{k}"] = (
                true_optimum / best_selected_latency
            )
        else:
            metrics[f"hit_{k}"] = np.nan
            metrics[f"regret_{k}"] = np.nan
            metrics[f"fraction_optimal_{k}"] = np.nan

        if n_good:
            good_hits = int(good_mask[selected].sum())
            metrics[f"precision_{k}"] = good_hits / selected_count
            metrics[f"recall_{k}"] = good_hits / n_good
        else:
            metrics[f"precision_{k}"] = np.nan
            metrics[f"recall_{k}"] = np.nan

    return metrics


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
        description="Evaluate the GEMM XGBoost pairwise-ranking model on the saved holdout split."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--out", default="rank_gemm_eval_report.txt")
    parser.add_argument(
        "--good_frac",
        type=float,
        default=0.10,
        help="Fraction of fastest valid configurations per workload defining the good region.",
    )
    args = parser.parse_args()

    if not 0 < args.good_frac <= 1:
        parser.error("--good_frac must be greater than 0 and at most 1.")

    # -------------------------------------------------------------------------
    # Load and preprocess data; use the existing saved split (no new split).
    # -------------------------------------------------------------------------
    raw_df = pd.read_csv(args.data)
    df = preprocess_data_gemm(raw_df)
    df["qid"] = df.groupby(["M", "N", "K"], sort=False).ngroup()

    with np.load(args.splits) as splits:
        development_indices = splits["development_indices"]
        holdout_indices = splits["holdout_indices"]

    train = df.iloc[development_indices].copy()
    test = df.iloc[holdout_indices].copy()

    # -------------------------------------------------------------------------
    # Load the trained rank model and predict ranking scores.
    # -------------------------------------------------------------------------
    bst = xgb.Booster()
    bst.load_model(args.model)

    features = list(bst.feature_names) if bst.feature_names else FEATURES.copy()
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise SystemExit(
            "Model expects features that are missing from the preprocessed data: "
            f"{missing_features}"
        )

    train["ranking_score"] = bst.predict(xgb.DMatrix(train[features]))
    test["ranking_score"] = bst.predict(xgb.DMatrix(test[features]))

    # -------------------------------------------------------------------------
    # Section 0: setup and sanity checks.
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
        map(tuple, train[["M", "N", "K"]].drop_duplicates().to_numpy())
    )
    test_keys = set(
        map(tuple, test[["M", "N", "K"]].drop_duplicates().to_numpy())
    )
    shared_groups = len(train_keys & test_keys)
    out(f"Workloads shared between train and test: {shared_groups} (should be 0)")
    out(
        f"Invalid configurations: train {100 * (~train['is_valid']).mean():.2f}% "
        f"| test {100 * (~test['is_valid']).mean():.2f}%"
    )
    out(f"Valid test configurations: {int(test['is_valid'].sum())}")
    out("Ranking direction: higher model score means a better configuration.")

    # -------------------------------------------------------------------------
    # Section 1: grouped and pooled Spearman correlation.
    # -------------------------------------------------------------------------
    header("1. SPEARMAN RANK CORRELATION")

    spearman_all = []
    spearman_valid = []
    for _, group in test.groupby("qid", sort=False):
        scores = group["ranking_score"].to_numpy(dtype=float)
        latencies = group["latency_ms"].to_numpy(dtype=float)
        valid = group["is_valid"].to_numpy(dtype=bool)

        # Positive correlation is good: high model score aligns with low latency.
        spearman_all.append(safe_spearman(scores, -latencies))
        if valid.sum() >= 3:
            spearman_valid.append(
                safe_spearman(scores[valid], -latencies[valid])
            )

    pooled_all = safe_spearman(
        test["ranking_score"].to_numpy(dtype=float),
        -test["latency_ms"].to_numpy(dtype=float),
    )
    test_valid = test[test["is_valid"]]
    pooled_valid = safe_spearman(
        test_valid["ranking_score"].to_numpy(dtype=float),
        -test_valid["latency_ms"].to_numpy(dtype=float),
    )

    out("Grouped Spearman (mean each workload equally):")
    out(f"  All configurations (invalid included): {summ(spearman_all)}")
    out(f"  Valid configurations only           : {summ(spearman_valid)}")
    out(f"Pooled Spearman, all configs (supplementary): {pooled_all:.4f}")
    out(f"Pooled Spearman, valid only (supplementary): {pooled_valid:.4f}")
    out(
        "Positive correlation is better. Pooled values mix workloads and can "
        "reflect differences in latency between matrix sizes; grouped values "
        "are the primary results."
    )

    # -------------------------------------------------------------------------
    # Section 2: pairwise ordering accuracy.
    # -------------------------------------------------------------------------
    header("2. PAIRWISE ACCURACY")

    pairwise_all = []
    pairwise_valid = []
    for _, group in test.groupby("qid", sort=False):
        latencies = group["latency_ms"].to_numpy(dtype=float)
        scores = group["ranking_score"].to_numpy(dtype=float)
        valid = group["is_valid"].to_numpy(dtype=bool)

        pairwise_all.append(pairwise_accuracy(latencies, scores))
        if valid.sum() >= 2:
            pairwise_valid.append(
                pairwise_accuracy(latencies[valid], scores[valid])
            )

    out("Unequal-latency pairs only; predicted ties receive 0.5 credit:")
    out(f"  All configurations (invalid included): {summ(pairwise_all)}")
    out(f"  Valid configurations only           : {summ(pairwise_valid)}")

    # -------------------------------------------------------------------------
    # Section 3: NDCG ranking quality.
    # -------------------------------------------------------------------------
    header("3. NDCG RANKING QUALITY")

    ndcg_all = {k: [] for k in list(NDCG_KS) + [None]}
    ndcg_valid = {k: [] for k in list(NDCG_KS) + [None]}

    for _, group in test.groupby("qid", sort=False):
        scores = group["ranking_score"].to_numpy(dtype=float)
        latencies = group["latency_ms"].to_numpy(dtype=float)
        valid = group["is_valid"].to_numpy(dtype=bool)
        relevance = relevance_labels(latencies)

        if len(group) >= 2 and valid.any():
            for k in list(NDCG_KS) + [None]:
                ndcg_all[k].append(
                    float(ndcg_score(relevance[None, :], scores[None, :], k=k))
                )

        if valid.sum() >= 2:
            valid_relevance = relevance[valid]
            valid_scores = scores[valid]
            for k in list(NDCG_KS) + [None]:
                ndcg_valid[k].append(
                    float(
                        ndcg_score(
                            valid_relevance[None, :], valid_scores[None, :], k=k
                        )
                    )
                )

    out(
        "Relevance: invalid = 0; valid = 1..30 by within-workload latency rank; "
        "linear gain."
    )
    for label, store in (
        ("All configurations (invalid included)", ndcg_all),
        ("Valid configurations only", ndcg_valid),
    ):
        out(f"{label}:")
        for k in list(NDCG_KS) + [None]:
            metric_name = f"NDCG@{k}" if k is not None else "NDCG (full)"
            out(f"  {metric_name:<12}: {summ(store[k])}")

    # -------------------------------------------------------------------------
    # Section 4: top-k hit rate, regret, and fraction of optimal.
    # -------------------------------------------------------------------------
    header("4. TOP-K SELECTION QUALITY")

    workload_results = []
    metric_names = [
        "spearman_all", "spearman_valid", "pairwise_all", "pairwise_valid",
        "ndcg_all", "ndcg_valid", "ndcg_at_10",
    ]
    for k in TOP_KS:
        metric_names.extend([
            f"hit_{k}", f"regret_{k}", f"fraction_optimal_{k}",
            f"precision_{k}", f"recall_{k}",
            f"invalid_fraction_{k}", f"has_invalid_{k}",
        ])

    aggregate = {name: [] for name in metric_names}

    for qid, group in test.groupby("qid", sort=False):
        metrics = calculate_workload_metrics(group, args.good_frac)
        for name in metric_names:
            aggregate[name].append(metrics.get(name, np.nan))

        workload_results.append({
            "qid": qid,
            "M": group["M"].iloc[0],
            "N": group["N"].iloc[0],
            "K": group["K"].iloc[0],
            "n_configs": len(group),
            **metrics,
        })

    out("Hit rate = fraction of workloads where the exact best measured config appears in top-k.")
    out("Regret@k = (best true latency in predicted top-k / true optimum) - 1.")
    out("Fraction optimal@k = true optimum / best true latency in predicted top-k.")
    for k in TOP_KS:
        out(f"Top-{k}:")
        out(f"  Hit rate             : {pct_summ(aggregate[f'hit_{k}'])}")
        out(f"  Regret               : {pct_summ(aggregate[f'regret_{k}'])}")
        out(f"  Fraction of optimal  : {pct_summ(aggregate[f'fraction_optimal_{k}'])}")

    # -------------------------------------------------------------------------
    # Section 5: good-region retrieval (precision and recall).
    # -------------------------------------------------------------------------
    header("5. GOOD-REGION RETRIEVAL")
    out(
        f"Good region = fastest {100 * args.good_frac:.1f}% of VALID configurations "
        "within each workload. Predicted top-k is selected from all configs."
    )
    out("Precision@k = good configs in predicted top-k / number selected.")
    out("Recall@k = good configs in predicted top-k / total good configs.")
    for k in TOP_KS:
        out(f"Top-{k}: Precision {pct_summ(aggregate[f'precision_{k}'])}")
        out(f"       Recall    {pct_summ(aggregate[f'recall_{k}'])}")

    # -------------------------------------------------------------------------
    # Section 6: invalid configurations selected in predicted top-k.
    # -------------------------------------------------------------------------
    header("6. INVALID CONFIGURATION SELECTION")
    out("Candidate fraction = mean invalid fraction among selected top-k configs.")
    out("Workload rate = fraction of workloads with at least one invalid in top-k.")
    for k in TOP_KS:
        out(f"Top-{k}:")
        out(
            f"  Invalid candidate fraction: "
            f"{pct_summ(aggregate[f'invalid_fraction_{k}'])}"
        )
        out(
            f"  Workloads with >=1 invalid: "
            f"{pct_summ(aggregate[f'has_invalid_{k}'])}"
        )

    # -------------------------------------------------------------------------
    # Section 7: per-workload metric distribution and worst workloads.
    # -------------------------------------------------------------------------
    header("7. PER-WORKLOAD RANKING PERFORMANCE")
    per_group_ndcg10 = [
        row["ndcg_at_10"] for row in workload_results
        if np.isfinite(row["ndcg_at_10"])
    ]
    per_group_spearman = [
        row["spearman_all"] for row in workload_results
        if np.isfinite(row["spearman_all"])
    ]
    per_group_pairwise = [
        row["pairwise_all"] for row in workload_results
        if np.isfinite(row["pairwise_all"])
    ]
    per_group_regret1 = [
        row["regret_1"] for row in workload_results
        if np.isfinite(row["regret_1"])
    ]

    out(f"NDCG@10       : {summ(per_group_ndcg10)}")
    out(f"Spearman      : {summ(per_group_spearman)}")
    out(f"Pairwise acc. : {summ(per_group_pairwise)}")
    out(f"Regret@1      : {pct_summ(per_group_regret1)}")
    out()

    eligible_worst = [
        row for row in workload_results if np.isfinite(row["ndcg_at_10"])
    ]
    if eligible_worst:
        out("Worst 5 workloads by NDCG@10:")
        out(
            f"{'(M, N, K)':<28} {'configs':>8} {'NDCG@10':>9} "
            f"{'Spearman':>10} {'Pairwise':>10} {'Regret@1':>11} {'Hit@5':>8}"
        )
        for row in sorted(eligible_worst, key=lambda item: item["ndcg_at_10"])[:5]:
            workload = f"({int(row['M'])}, {int(row['N'])}, {int(row['K'])})"
            regret_text = (
                f"{100 * row['regret_1']:.2f}%"
                if np.isfinite(row["regret_1"]) else "n/a"
            )
            spearman_text = (
                f"{row['spearman_all']:.4f}"
                if np.isfinite(row["spearman_all"]) else "n/a"
            )
            pairwise_text = (
                f"{row['pairwise_all']:.4f}"
                if np.isfinite(row["pairwise_all"]) else "n/a"
            )
            hit5_text = (
                f"{row['hit_5']:.0f}"
                if np.isfinite(row["hit_5"]) else "n/a"
            )
            out(
                f"{workload:<28} {row['n_configs']:8d} "
                f"{row['ndcg_at_10']:9.4f} {spearman_text:>10} "
                f"{pairwise_text:>10} {regret_text:>11} {hit5_text:>8}"
            )
    else:
        out("No eligible workloads for NDCG@10 worst-workload analysis.")

    # -------------------------------------------------------------------------
    # Section 8: inference latency.
    # -------------------------------------------------------------------------
    header("8. INFERENCE LATENCY (XGBOOST RANKING PREDICTION)")
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
    if check_count:
        dm_check = xgb.DMatrix(X[:check_count], feature_names=list(features))
        numpy_predictions = bst.predict(dm_check)
        dataframe_predictions = test["ranking_score"].to_numpy()[:check_count]
        max_difference = float(
            np.max(np.abs(numpy_predictions - dataframe_predictions))
        )
    else:
        max_difference = 0.0

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
    # Write text report.
    # -------------------------------------------------------------------------
    output_directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(LINES) + "\n")

    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
