
import pandas as pd
import numpy as np
import xgboost as xgb
import json

from sklearn.metrics import mean_squared_error


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42

DATASET_PATH = "../datasets/t4/bmm/bmm_dataset.csv"

PARAMS_PATH = "./hyperparameter-tuning/xgb_logexe_best_params_bmm.json"
SPLITS_PATH = "./hyperparameter-tuning/bmm_group_cv_splits.npz"

MODEL_PATH = "xgb_logexe_model_bmm.json"


# ==========================================
# 1. Preprocessing
# ==========================================

def preprocess_data_bmm(raw_df):

    keep_cols = [
        "BATCH_SIZE", "M", "N", "K",
        "TILE_M", "TILE_N", "TILE_K",
        "THREAD_X", "THREAD_Y", "THREAD_Z",
        "UNROLL_FACTOR",
        "latency_ms"
    ]

    df = raw_df[keep_cols].copy()

    # Convert invalid configurations to a massive latency penalty
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = 1e8

    # Log transformation
    df["log_latency"] = np.log(df["latency_ms"])

    # ======================================
    # Derived Features
    # ======================================

    df["TOTAL_THREADS"] = (
        df["THREAD_X"] *
        df["THREAD_Y"] *
        df["THREAD_Z"]
    )

    # Grid size includes batch size
    df["GRID_SIZE"] = (
        np.ceil(df["M"] / df["TILE_M"]) *
        np.ceil(df["N"] / df["TILE_N"]) *
        df["BATCH_SIZE"]
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

    # Group ID
    df["qid"] = df.groupby(
        ["BATCH_SIZE", "M", "N", "K"],
        sort=False
    ).ngroup()

    return df


# ==========================================
# 2. Main Execution
# ==========================================

if __name__ == "__main__":

    # ======================================
    # Load and Preprocess Data
    # ======================================

    raw_df = pd.read_csv(DATASET_PATH)

    df = preprocess_data_bmm(raw_df)

    features = [
        "BATCH_SIZE", "M", "N", "K",
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

    train_data = df.iloc[dev_idx]
    test_data = df.iloc[holdout_idx]

    print(
        f"Training on {len(train_data)} configurations "
        f"across {train_data['qid'].nunique()} workload sizes."
    )

    print(
        f"Testing on {len(test_data)} configurations "
        f"across {test_data['qid'].nunique()} workload sizes."
    )

    # ======================================
    # Create DMatrices
    # ======================================

    dtrain = xgb.DMatrix(
        train_data[features],
        label=train_data["log_latency"]
    )

    dtest = xgb.DMatrix(
        test_data[features],
        label=test_data["log_latency"]
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
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": RANDOM_STATE
    })

    # ======================================
    # Train Final Model
    # ======================================

    print("\nTraining final XGBoost Log-Regression model...")

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

    preds_log = bst.predict(dtest)

    # RMSE in log space
    test_rmse_log = np.sqrt(
        mean_squared_error(
            test_data["log_latency"],
            preds_log
        )
    )

    print(
        f"\nFinal Test RMSE (Log Space): "
        f"{test_rmse_log:.4f}"
    )

    # ======================================
    # Evaluate in Milliseconds
    # ======================================

    preds_ms = np.exp(preds_log)

    # Exclude invalid configurations
    valid_mask = test_data["latency_ms"] < 1e7

    if valid_mask.sum() > 0:

        test_rmse_ms = np.sqrt(
            mean_squared_error(
                test_data.loc[valid_mask, "latency_ms"],
                preds_ms[valid_mask]
            )
        )

        print(
            f"Final Test RMSE (Real Milliseconds, "
            f"valid configs only): {test_rmse_ms:.4f} ms"
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
