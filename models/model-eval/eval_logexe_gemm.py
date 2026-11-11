#!/usr/bin/env python3
"""
Evaluate the XGBoost log-latency GEMM surrogate (reg:squarederror on log(latency_ms)).

Reproduces the training split (GroupShuffleSplit, test_size=0.2, random_state=42,
groups = (M, N, K)) and evaluates the saved model on the held-out 20%.

Usage (from the same folder you train in):
    python evaluate_logexe_gemm.py
    python evaluate_logexe_gemm.py --threshold 12.5
    python evaluate_logexe_gemm.py --test_csv ../datasets/t4/gemm/fresh_gemm_dataset.csv

Writes a text report (default: logexe_gemm_eval_report.txt).
"""
import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, confusion_matrix, ndcg_score,
                             precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

INVALID_LAT = 1e8       # penalty latency used in training
VALID_CUTOFF = 1e7      # latency_ms >= this  ->  invalid config
LOG_INVALID = np.log(INVALID_LAT)

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


# ----------------------------------------------------------------------------
# Preprocessing (identical to the training script)
# ----------------------------------------------------------------------------
def preprocess_data_gemm(raw_df):
    keep_cols = ["M", "N", "K", "TILE_M", "TILE_N", "TILE_K",
                 "THREAD_X", "THREAD_Y", "THREAD_Z", "UNROLL_FACTOR", "latency_ms"]
    df = raw_df[keep_cols].copy()
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = INVALID_LAT
    df["log_latency"] = np.log(df["latency_ms"])

    df["TOTAL_THREADS"] = df["THREAD_X"] * df["THREAD_Y"] * df["THREAD_Z"]
    df["GRID_SIZE"] = np.ceil(df["M"] / df["TILE_M"]) * np.ceil(df["N"] / df["TILE_N"])
    df["WORK_PER_THREAD_M"] = (df["TILE_M"] + df["THREAD_Y"] - 1) // df["THREAD_Y"]
    df["WORK_PER_THREAD_N"] = (df["TILE_N"] + df["THREAD_X"] - 1) // df["THREAD_X"]
    df["SHMEM_FOOTPRINT"] = (df["TILE_M"] * df["TILE_K"]) + (df["TILE_K"] * df["TILE_N"])
    df["UNROLL_ALIGNMENT"] = (df["TILE_K"] % df["UNROLL_FACTOR"] == 0).astype(int)
    df["BOUNDARY_ALIGNMENT_M"] = (df["M"] % df["TILE_M"] == 0).astype(int)
    df["BOUNDARY_ALIGNMENT_N"] = (df["N"] % df["TILE_N"] == 0).astype(int)
    return df


# ----------------------------------------------------------------------------
# Report helper
# ----------------------------------------------------------------------------
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def header(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def reg_metrics(t_log, p_log):
    """Regression metrics. RMSE/R2(log) in log space; RPE/MAPE/R2(ms) in real ms."""
    t_log = np.asarray(t_log, dtype=float)
    p_log = np.asarray(p_log, dtype=float)
    if len(t_log) == 0:
        return None
    err = p_log - t_log
    rpe = np.abs(np.expm1(err))             # |exp(p) - exp(t)| / exp(t)
    ss_tot = np.sum((t_log - t_log.mean()) ** 2)
    return {
        "n": len(t_log),
        "log_rmse": float(np.sqrt(np.mean(err ** 2))),
        "log_bias": float(err.mean()),       # >0 pessimistic, <0 optimistic
        "median_rpe": float(np.median(rpe)),
        "mape": float(rpe.mean()),
        "r2_log": float(r2_score(t_log, p_log)) if ss_tot > 0 else float("nan"),
        "r2_ms": float(r2_score(np.exp(t_log), np.exp(p_log))) if ss_tot > 0 else float("nan"),
    }


def show_reg(title, m):
    if m is None:
        out(f"{title}: no samples")
        return
    out(f"{title}  (n = {m['n']})")
    out(f"  Log-space RMSE          : {m['log_rmse']:.4f}")
    out(f"  Mean log error (bias)   : {m['log_bias']:+.4f}   (negative = model optimistic)")
    out(f"  Median RPE              : {100 * m['median_rpe']:.2f} %")
    out(f"  MAPE                    : {100 * m['mape']:.2f} %")
    out(f"  R^2 (log space)         : {m['r2_log']:.4f}")
    out(f"  R^2 (real ms)           : {m['r2_ms']:.4f}")


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def summ(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "n/a (no usable groups)"
    return (f"mean {v.mean():.4f} | median {np.median(v):.4f} | std {v.std():.4f} "
            f"| min {v.min():.4f} | groups {len(v)}")


def pct_summ(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "n/a"
    return (f"mean {v.mean():.4f} | median {np.median(v):.4f} | "
            f"5th pct {np.percentile(v, 5):.4f} | min {v.min():.4f}")


def relevance_labels(lat):
    """
    Relevance grades for NDCG (within one matrix size):
      invalid config -> 0
      valid configs  -> 1..30, linear in the latency rank (fastest = 30, slowest = 1)
    Linear gain is used (matches ndcg_exp_gain=False in the rank model).
    """
    lat = np.asarray(lat, dtype=float)
    valid = lat < VALID_CUTOFF
    rel = np.zeros(len(lat))
    nv = valid.sum()
    if nv >= 1:
        ranks = pd.Series(lat[valid]).rank(method="average").values - 1.0   # 0 = fastest
        pct = ranks / max(nv - 1, 1)
        rel[valid] = 1 + np.round(29 * (1 - pct))
    return rel


# ----------------------------------------------------------------------------
# Timing helper
# ----------------------------------------------------------------------------
def bench(fn, repeats, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 95))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../datasets/t4/gemm/gemm_dataset.csv")
    ap.add_argument("--model", default="../xgb_logexe_model_gemm.json")
    ap.add_argument("--out", default="logexe_gemm_eval_report.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=None,
                    help="log-latency threshold above which a config is predicted invalid "
                         "(default: midpoint between the slowest valid TRAIN config and log(1e8))")
    ap.add_argument("--good_frac", type=float, default=0.10,
                    help="fraction of fastest valid configs per matrix size = 'good region'")
    ap.add_argument("--test_csv", default=None,
                    help="optional fresh dataset (unseen matrix sizes) to evaluate on instead of the 20%% split")
    args = ap.parse_args()

    # ---------------- Load data and reproduce the split ----------------
    df = preprocess_data_gemm(pd.read_csv(args.data))
    df["qid"] = df.groupby(["M", "N", "K"]).ngroup()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(df, groups=df["qid"]))
    train = df.iloc[tr_idx].copy()
    test = df.iloc[te_idx].copy()

    eval_source = "20% group-held-out split of " + args.data
    if args.test_csv:
        test = preprocess_data_gemm(pd.read_csv(args.test_csv))
        test["qid"] = test.groupby(["M", "N", "K"]).ngroup()
        eval_source = "fresh dataset " + args.test_csv

    # ---------------- Load model ----------------
    bst = xgb.Booster()
    bst.load_model(args.model)
    features = bst.feature_names if bst.feature_names else FEATURES
    missing = [f for f in features if f not in test.columns]
    if missing:
        raise SystemExit(f"Model expects features not in the data: {missing}")

    test["pred_log"] = bst.predict(xgb.DMatrix(test[features]))
    train["pred_log"] = bst.predict(xgb.DMatrix(train[features]))
    for d in (train, test):
        d["valid"] = d["latency_ms"] < VALID_CUTOFF

    # ---------------- Section 0: setup ----------------
    header("0. SETUP AND SANITY CHECKS")
    out(f"Model file          : {args.model}")
    out(f"Boosted rounds      : {bst.num_boosted_rounds()}")
    out(f"Evaluated on        : {eval_source}")
    out(f"Split seed          : {args.seed}")
    out(f"Train               : {len(train)} configs, {train['qid'].nunique()} matrix sizes")
    out(f"Test                : {len(test)} configs, {test['qid'].nunique()} matrix sizes")
    out("  -> Compare the train/test counts above with the lines printed by your "
        "training script;\n     if they differ, the split is NOT the same as the one used for training.")
    train_keys = set(map(tuple, train[["M", "N", "K"]].drop_duplicates().values))
    test_keys = set(map(tuple, test[["M", "N", "K"]].drop_duplicates().values))
    out(f"Matrix sizes shared between train and test: {len(train_keys & test_keys)} "
        f"(should be 0)")
    out(f"Invalid configs     : train {100 * (~train['valid']).mean():.2f} % | "
        f"test {100 * (~test['valid']).mean():.2f} %")
    out(f"Valid test configs  : {int(test['valid'].sum())}")

    tv = test[test["valid"]]

    # ---------------- Section 1: regression metrics (valid only) ----------------
    header("1. REGRESSION METRICS - VALID CONFIGS ONLY")
    m_test = reg_metrics(tv["log_latency"], tv["pred_log"])
    show_reg("TEST (held-out)", m_test)
    out()
    trv = train[train["valid"]]
    m_train = reg_metrics(trv["log_latency"], trv["pred_log"])
    out(f"For reference, TRAIN log-RMSE (valid only): {m_train['log_rmse']:.4f}   "
        f"-> train/test gap = {m_test['log_rmse'] - m_train['log_rmse']:+.4f}")
    out("Note: RPE/MAPE are computed on real milliseconds (exp of predictions); "
        "log-RMSE and R^2(log) in log space.")

    # ---------------- Section 2: good region ----------------
    header(f"2. ACCURACY IN THE GOOD REGION (fastest {100 * args.good_frac:.0f} % valid configs per matrix size)")
    good_mask = np.zeros(len(tv), dtype=bool)
    tv = tv.reset_index(drop=True)
    for _, idx in tv.groupby("qid").indices.items():
        n_good = max(1, int(np.ceil(args.good_frac * len(idx))))
        order = idx[np.argsort(tv["latency_ms"].values[idx], kind="stable")]
        good_mask[order[:n_good]] = True
    tg = tv[good_mask]
    show_reg("TEST, good region", reg_metrics(tg["log_latency"], tg["pred_log"]))
    out("Note: R^2 is low/negative here by construction (the true values span a narrow range); "
        "rely on RMSE, RPE and bias.")

    # ---------------- Section 3: per matrix size spread ----------------
    header("3. ERROR SPREAD ACROSS HELD-OUT MATRIX SIZES (valid configs only)")
    per_group = []
    for qid, g in tv.groupby("qid"):
        e = g["pred_log"].values - g["log_latency"].values
        per_group.append((qid, g["M"].iloc[0], g["N"].iloc[0], g["K"].iloc[0],
                          float(np.sqrt(np.mean(e ** 2))), len(g)))
    rm = np.array([p[4] for p in per_group])
    out(f"Per-size log-RMSE: mean {rm.mean():.4f} | median {np.median(rm):.4f} | "
        f"std {rm.std():.4f} | min {rm.min():.4f} | max {rm.max():.4f}  ({len(rm)} sizes)")
    out("Worst 5 matrix sizes (M, N, K, log-RMSE, #valid configs):")
    for _, M, N, K, r, n in sorted(per_group, key=lambda p: -p[4])[:5]:
        out(f"  {int(M)} x {int(N)} x {int(K)} : {r:.4f}   (n={n})")

    # ---------------- Section 4: invalid detection ----------------
    header("4. INVALID-CONFIG DETECTION (threshold on predicted log-latency)")
    valid_train_max = trv["log_latency"].max()
    thr = args.threshold if args.threshold is not None else (valid_train_max + LOG_INVALID) / 2
    out(f"Slowest valid TRAIN config (log) : {valid_train_max:.3f}")
    out(f"Invalid penalty (log)            : {LOG_INVALID:.3f}")
    out(f"Threshold used                   : {thr:.3f}  "
        f"({'user-given' if args.threshold is not None else 'midpoint of the two clusters, from train labels only'})")
    inv_true = (~test["valid"]).values
    score = test["pred_log"].values
    if inv_true.sum() == 0 or inv_true.all():
        out("Test set has only one class (all valid or all invalid) - detection metrics skipped.")
    else:
        out(f"AUC (threshold-free)             : {roc_auc_score(inv_true, score):.4f}")
        out(f"Average precision (PR-AUC)       : {average_precision_score(inv_true, score):.4f}")
        out()

        def det_row(th):
            pred_inv = score >= th
            p, r, f1, _ = precision_recall_fscore_support(
                inv_true, pred_inv, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(inv_true, pred_inv, labels=[False, True]).ravel()
            acc = (tp + tn) / len(inv_true)
            return p, r, f1, acc, tp, fp, fn, tn

        p, r, f1, acc, tp, fp, fn, tn = det_row(thr)
        out(f"At the chosen threshold ({thr:.3f}):")
        out(f"  Precision : {p:.4f}")
        out(f"  Recall    : {r:.4f}   <- invalid configs that would be caught")
        out(f"  F1        : {f1:.4f}")
        out(f"  Accuracy  : {acc:.4f}")
        out(f"  Confusion : TP {tp} | FP {fp} | FN {fn} | TN {tn}")
        out(f"  FN = invalid configs predicted valid (the dangerous ones for PSO): "
            f"{fn} of {tp + fn} ({100 * fn / max(tp + fn, 1):.2f} %)")
        out()
        out("Threshold sweep (sanity check on the chosen cutoff):")
        out(f"  {'thr':>7} {'prec':>7} {'recall':>7} {'F1':>7}")
        for th in np.linspace(valid_train_max, LOG_INVALID - 0.5, 6):
            p_, r_, f_, *_ = det_row(th)
            out(f"  {th:7.2f} {p_:7.4f} {r_:7.4f} {f_:7.4f}")

    # ---------------- Section 5: ranking metrics ----------------
    header("5. RANKING METRICS ON THE LOG-EXE PREDICTIONS (per matrix size, then averaged)")
    ks = [5, 10, 20]
    sp_all, sp_val = [], []
    ndcg_all = {k: [] for k in ks + [None]}
    ndcg_val = {k: [] for k in ks + [None]}
    frac_opt = {k: [] for k in (1, 5, 10)}
    top1_invalid = []

    for qid, g in test.groupby("qid"):
        pred = g["pred_log"].values
        lat = g["latency_ms"].values
        valid = lat < VALID_CUTOFF

        sp_all.append(safe_spearman(pred, lat))
        if valid.sum() >= 3:
            sp_val.append(safe_spearman(pred[valid], lat[valid]))

        rel = relevance_labels(lat)
        if len(g) >= 2 and valid.any():
            for k in ks + [None]:
                ndcg_all[k].append(ndcg_score(rel[None, :], -pred[None, :], k=k))
        if valid.sum() >= 2:
            for k in ks + [None]:
                ndcg_val[k].append(ndcg_score(rel[valid][None, :], -pred[valid][None, :], k=k))

        if valid.any():
            opt = lat[valid].min()
            order = np.argsort(pred, kind="stable")
            for k in frac_opt:
                frac_opt[k].append(opt / lat[order[:k]].min())   # ~0 if top-k are all invalid
            top1_invalid.append(float(not valid[order[0]]))

    out("Spearman correlation (predicted vs true latency; lower latency = better, so + is good):")
    out(f"  Grouped, all configs (invalid tied at 1e8) : {summ(sp_all)}")
    out(f"  Grouped, valid configs only                : {summ(sp_val)}")
    gs = test[["pred_log", "latency_ms"]]
    out(f"  Pooled global, all configs                 : "
        f"{safe_spearman(gs['pred_log'], gs['latency_ms']):.4f}")
    out(f"  Pooled global, valid configs only          : "
        f"{safe_spearman(tv['pred_log'], tv['latency_ms']):.4f}")
    out("  (Pooled values mix different matrix sizes, so they partly measure 'big matrix = slow'; "
        "the grouped numbers are the meaningful ones.)")
    out()
    out("NDCG (relevance: invalid = 0, valid = 1..30 linear in latency rank within the matrix size):")
    for label, store in (("all configs (invalid included)", ndcg_all), ("valid configs only", ndcg_val)):
        out(f"  {label}:")
        for k in ks + [None]:
            name = "NDCG@%d" % k if k else "NDCG (full)"
            out(f"    {name:<12}: {summ(store[k])}")
    out()
    out("Top-k fraction of optimal (best TRUE latency among the model's top-k / true optimum; 1.0 = found best):")
    for k in frac_opt:
        out(f"  top-{k:<2}: {pct_summ(frac_opt[k])}")
    out(f"  Share of matrix sizes where the model's top-1 pick is an INVALID config: "
        f"{100 * np.mean(top1_invalid):.2f} %")

    # ---------------- Section 6: inference latency ----------------
    header("6. INFERENCE LATENCY (raw XGBoost predict)")
    out(f"xgboost {xgb.__version__} | python {platform.python_version()} | "
        f"CPU logical cores: {os.cpu_count()} | {platform.processor() or platform.machine()}")
    out("Booster device/threads are whatever your XGBoost default is (CPU, all cores unless set).")
    X = test[features].to_numpy(dtype=np.float32)

    dm_check = xgb.DMatrix(X[:256], feature_names=list(features))
    diff = np.max(np.abs(bst.predict(dm_check) - test["pred_log"].values[:256]))
    out(f"Sanity: numpy-DMatrix vs DataFrame-DMatrix predictions, max abs diff = {diff:.2e}")
    out()
    out(f"  {'batch':>7} | {'predict only':^31} | {'DMatrix build + predict':^31}")
    out(f"  {'':>7} | {'ms/call':>9} {'us/config':>9} {'configs/s':>10} | "
        f"{'ms/call':>9} {'us/config':>9} {'configs/s':>10}")
    for b in (1, 32, 128, 1024, len(X)):
        idx = np.resize(np.arange(len(X)), b)
        Xb = X[idx]
        dm = xgb.DMatrix(Xb, feature_names=list(features))
        reps = 300 if b == 1 else (100 if b <= 128 else 20)
        t_pred, _ = bench(lambda: bst.predict(dm), reps)
        t_full, _ = bench(lambda: bst.predict(xgb.DMatrix(Xb, feature_names=list(features))), reps)
        out(f"  {b:>7} | {1e3 * t_pred:9.3f} {1e6 * t_pred / b:9.2f} {b / t_pred:10.0f} | "
            f"{1e3 * t_full:9.3f} {1e6 * t_full / b:9.2f} {b / t_full:10.0f}")
    out("Median over repeated calls after warm-up. Batch ~ PSO swarm size is the realistic case; "
        "'DMatrix build + predict' is what a PSO loop actually pays per iteration.")

    # ---------------- Write report ----------------
    with open(args.out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()