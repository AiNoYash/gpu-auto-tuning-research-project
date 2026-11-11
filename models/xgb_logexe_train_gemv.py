
import pandas as pd
import numpy as np
import xgboost as xgb
import json

from sklearn.metrics import mean_squared_error


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42

DATASET_PATH = "../datasets/t4/gemv/gemv_dataset.csv"

PARAMS_PATH = "./hyperparameter-tuning/xgb_logexe_best_params_gemv.json"
SPLITS_PATH = "./hyperparameter-tuning/gemv_group_cv_splits.npz"

MODEL_PATH = "xgb_logexe_model_gemv.json"


# ==========================================
# 1. Preprocessing
# ==========================================

def preprocess_data_gemv_log(raw_df):

    keep_cols = [
        "M", "K",
        "BLOCK_DIM_X", "ROWS_PER_BLOCK",
        "TILE_K", "UNROLL_FACTOR",
        "latency_ms"
    ]

    df = raw_df[keep_cols].copy()

    # Convert invalid configurations to a massive latency penalty
    df.loc[df["latency_ms"] == -1.0, "latency_ms"] = 1e8

    # Log transformation
    df["log_latency"] = np.log(df["latency_ms"])

    # ======================================
    # Derived Features for GEMV
    # ======================================

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

    # Group ID
    df["qid"] = df.groupby(
        ["M", "K"],
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

    df = preprocess_data_gemv_log(raw_df)

    features = [
        "M", "K",
        "BLOCK_DIM_X", "ROWS_PER_BLOCK",
        "TILE_K", "UNROLL_FACTOR",
        "GRID_SIZE",
        "SHMEM_ELEMENTS",
        "WORK_PER_THREAD_K",
        "TILE_K_ALIGNED",
        "UNROLL_ALIGNMENT",
        "BOUNDARY_ALIGNMENT_M",
        "BOUNDARY_ALIGNMENT_K"
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

    print("\nTraining final XGBoost Log-Regression model for GEMV...")

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
    # Save Final Model for PSO
    # ======================================

    bst.save_model(MODEL_PATH)

    print(
        f"\nModel successfully saved to "
        f"{MODEL_PATH} for PSO inference."
    )

    print("\nFinal training completed.")
