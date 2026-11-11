
import pandas as pd
import numpy as np
import xgboost as xgb
import json

from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.metrics import ndcg_score
import warnings

warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42

DATASET_PATH = "../datasets/t4/gemm/gemm_dataset.csv"

PARAMS_PATH = "./hyperparameter-tuning/xgb_rank_best_params_gemm.json"
SPLITS_PATH = "./hyperparameter-tuning/gemm_group_cv_splits.npz"

MODEL_PATH = "xgb_rank_model_gemm.json"


# ==========================================
# 1. Preprocessing
# ==========================================

def preprocess_data_gemm(raw_df):

    keep_cols = [
        "M", "N", "K",
        "TILE_M", "TILE_N", "TILE_K",
        "THREAD_X", "THREAD_Y", "THREAD_Z",
        "UNROLL_FACTOR",
        "latency_ms"
    ]

    df = raw_df[keep_cols].copy()

    # Convert invalid configurations to a massive latency penalty
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = 1e8

    # ======================================
    # Derived Features
    # ======================================

    df["TOTAL_THREADS"] = (
        df["THREAD_X"] *
        df["THREAD_Y"] *
        df["THREAD_Z"]
    )

    df["GRID_SIZE"] = (
        np.ceil(df["M"] / df["TILE_M"]) *
        np.ceil(df["N"] / df["TILE_N"])
    )

    df["WORK_PER_THREAD_M"] = (
        (df["TILE_M"] + df["THREAD_Y"] - 1)
        // df["THREAD_Y"]
    )

    df["WORK_PER_THREAD_N"] = (
        (df["TILE_N"] + df["THREAD_X"] - 1)
        // df["THREAD_X"]
    )

    df["SHMEM_FOOTPRINT"] = (
        (df["TILE_M"] * df["TILE_K"]) +
        (df["TILE_K"] * df["TILE_N"])
    )

    df["UNROLL_ALIGNMENT"] = (
        df["TILE_K"] % df["UNROLL_FACTOR"] == 0
    ).astype(int)

    df["BOUNDARY_ALIGNMENT_M"] = (
        df["M"] % df["TILE_M"] == 0
    ).astype(int)

    df["BOUNDARY_ALIGNMENT_N"] = (
        df["N"] % df["TILE_N"] == 0
    ).astype(int)

    # ======================================
    # Ranking Targets and Grouping
    # ======================================

    df["qid"] = df.groupby(
        ["M", "N", "K"],
        sort=False
    ).ngroup()

    # Highest latency = rank 1
    # Lowest latency = highest rank

    df["target"] = (
        df.groupby("qid")["latency_ms"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    return df


# ==========================================
# 2. Custom Grouped Spearman Metric
# ==========================================

def calculate_grouped_spearman(y_true, y_pred, qids):

    df_eval = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "qid": qids
    })

    spearman_scores = []

    for _, group in df_eval.groupby("qid"):

        if len(group) > 1:

            preds = group["pred"].values

            rho, _ = spearmanr(
                group["true"].values,
                preds
            )

            if np.isnan(rho):
                if group["true"].nunique() > 1:
                    spearman_scores.append(0.0)
            else:
                spearman_scores.append(rho)

    return (
        np.mean(spearman_scores)
        if spearman_scores else 0.0
    )


# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":

    # ======================================
    # Load and Preprocess Data
    # ======================================

    raw_df = pd.read_csv(DATASET_PATH)

    df = preprocess_data_gemm(raw_df)

    features = [
        "M", "N", "K",
        "TILE_M", "TILE_N", "TILE_K",
        "THREAD_X", "THREAD_Y", "THREAD_Z",
        "UNROLL_FACTOR",
        "TOTAL_THREADS",
        "GRID_SIZE",
        "WORK_PER_THREAD_M",
        "WORK_PER_THREAD_N",
        "SHMEM_FOOTPRINT",
        "UNROLL_ALIGNMENT",
        "BOUNDARY_ALIGNMENT_M",
        "BOUNDARY_ALIGNMENT_N"
    ]

    # ======================================
    # Load Existing 80:20 Split
    # ======================================

    splits = np.load(SPLITS_PATH)

    dev_idx = splits["development_indices"]
    holdout_idx = splits["holdout_indices"]

    train_data = df.iloc[dev_idx].copy()
    test_data = df.iloc[holdout_idx].copy()

    # XGBoost ranking requires data sorted by qid
    train_data = train_data.sort_values("qid").reset_index(drop=True)
    test_data = test_data.sort_values("qid").reset_index(drop=True)

    print(
        f"Training on {len(train_data)} configurations "
        f"across {train_data['qid'].nunique()} matrix sizes."
    )

    print(
        f"Testing on {len(test_data)} configurations "
        f"across {test_data['qid'].nunique()} matrix sizes."
    )

    # ======================================
    # Create DMatrices
    # ======================================

    dtrain = xgb.DMatrix(
        train_data[features],
        label=train_data["target"],
        qid=train_data["qid"]
    )

    dtest = xgb.DMatrix(
        test_data[features],
        label=test_data["target"],
        qid=test_data["qid"]
    )

    # ======================================
    # Load Hyperparameters
    # ======================================

    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    # Extract final boosting rounds
    num_boost_round = int(best_params.pop("n_estimators"))

    # Re-inject fixed structural parameters
    best_params.update({
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "seed": RANDOM_STATE
    })

    # ======================================
    # Train Final Model
    # ======================================

    print("\nTraining final XGBoost Pairwise Ranking model...")

    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train")],
        verbose_eval=50
    )

    # ======================================
    # Evaluate on Holdout Test Set
    # ======================================

    preds = bst.predict(dtest)

    test_spearman = calculate_grouped_spearman(
        test_data["target"].values,
        preds,
        test_data["qid"].values
    )

    print(
        f"\nFinal Test Spearman Correlation: "
        f"{test_spearman:.4f}"
    )

    # ======================================
    # Save Final Model for PSO
    # ======================================

    bst.save_model(MODEL_PATH)

    print(
        f"\nModel successfully saved to "
        f"{MODEL_PATH} for PSO inference."
    )

    print("\nFinal training completed.")
