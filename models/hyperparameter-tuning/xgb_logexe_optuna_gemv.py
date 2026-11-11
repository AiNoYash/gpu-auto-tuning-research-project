
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json

from sklearn.model_selection import GroupShuffleSplit, GroupKFold


# ==========================================
# Configuration
# ==========================================

RANDOM_STATE = 42

N_SPLITS = 5
N_TRIALS = 300

MAX_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50

# Increase final number of estimators by 10%
FINAL_ESTIMATOR_MULTIPLIER = 1.1

DATASET_PATH = "../../datasets/t4/gemv/gemv_dataset.csv"

PARAMS_OUTPUT = "xgb_logexe_best_params_gemv.json"
CV_SUMMARY_OUTPUT = "xgb_logexe_cv_summary_gemv.json"
SPLITS_OUTPUT = "gemv_group_cv_splits.npz"


FEATURES = [
    "M", "K",
    "BLOCK_DIM_X", "ROWS_PER_BLOCK", "TILE_K", "UNROLL_FACTOR",
    "GRID_SIZE",
    "SHMEM_ELEMENTS",
    "WORK_PER_THREAD_K",
    "TILE_K_ALIGNED",
    "UNROLL_ALIGNMENT",
    "BOUNDARY_ALIGNMENT_M",
    "BOUNDARY_ALIGNMENT_K"
]


# ==========================================
# 1. Preprocessing
# ==========================================

def preprocess_data_gemv_log(raw_df):

    keep_cols = [
        "M", "K",
        "BLOCK_DIM_X", "ROWS_PER_BLOCK", "TILE_K",
        "UNROLL_FACTOR",
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

    # Group ID: each unique GEMV workload gets one qid
    df["qid"] = df.groupby(
        ["M", "K"],
        sort=False
    ).ngroup()


    return df


# ==========================================
# 2. Create Outer Split + Inner CV Folds
# ==========================================

def create_data_splits(df):

    groups = df["qid"].values

    # Outer 80:20 group-aware split
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    dev_idx, holdout_idx = next(
        gss.split(df, groups=groups)
    )

    # 80% development data
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    # 20% untouched holdout data
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)

    # Inner 5-fold GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)

    cv_splits = list(
        gkf.split(
            dev_df[FEATURES],
            dev_df["log_latency"],
            groups=dev_df["qid"]
        )
    )

    return (
        dev_df,
        holdout_df,
        dev_idx,
        holdout_idx,
        cv_splits
    )


# ==========================================
# 3. Optuna Objective Function
# ==========================================

def objective_gemv_log(trial, dev_df, cv_splits):

    # Hyperparameter search space
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.05, log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),

        "gamma": trial.suggest_float(
            "gamma", 0.0, 5.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-3, 10.0, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),

        "seed": RANDOM_STATE,
        "tree_method": "hist",
        "verbosity": 0
    }

    fold_rmses = []
    fold_best_rounds = []

    # ======================================
    # Run all 5 CV folds
    # ======================================

    for fold, (train_idx, val_idx) in enumerate(cv_splits):

        train_data = dev_df.iloc[train_idx]
        val_data = dev_df.iloc[val_idx]

        # Create XGBoost DMatrices
        dtrain = xgb.DMatrix(
            train_data[FEATURES],
            label=train_data["log_latency"]
        )

        dval = xgb.DMatrix(
            val_data[FEATURES],
            label=val_data["log_latency"]
        )

        # Train with early stopping independently on this fold
        bst = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

        # Best validation RMSE for this fold
        fold_rmse = float(bst.best_score)

        fold_rmses.append(fold_rmse)

        # Best iteration is zero-based
        best_rounds = int(bst.best_iteration) + 1

        fold_best_rounds.append(best_rounds)

        # ==================================
        # Optuna Fold-level Pruning
        # ==================================

        # Running mean validation RMSE
        running_mean_rmse = float(np.mean(fold_rmses))

        # Report intermediate result to Optuna
        trial.report(running_mean_rmse, step=fold)

        # Check whether this trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ======================================
    # Aggregate CV Results
    # ======================================

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))

    # Store useful information for the best trial
    trial.set_user_attr(
        "fold_rmses",
        fold_rmses
    )

    trial.set_user_attr(
        "fold_best_rounds",
        fold_best_rounds
    )

    trial.set_user_attr(
        "mean_rmse",
        mean_rmse
    )

    trial.set_user_attr(
        "std_rmse",
        std_rmse
    )

    # Optuna minimizes mean validation RMSE
    return mean_rmse


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":

    # Load dataset
    raw_df = pd.read_csv(DATASET_PATH)

    # Preprocessing
    df = preprocess_data_gemv_log(raw_df)

    # Outer split + inner CV
    (
        dev_df,
        holdout_df,
        dev_idx,
        holdout_idx,
        cv_splits
    ) = create_data_splits(df)

    print("\n========== DATA SPLIT ==========")

    print(f"Total rows: {len(df)}")
    print(f"Development rows: {len(dev_df)}")
    print(f"Holdout rows: {len(holdout_df)}")

    print(
        f"Development workload groups: "
        f"{dev_df['qid'].nunique()}"
    )

    print(
        f"Holdout workload groups: "
        f"{holdout_df['qid'].nunique()}"
    )

    # Save split indices so the ranking model can reuse
    # exactly the same outer split and inner CV folds.
    split_data = {
        "development_indices": dev_idx,
        "holdout_indices": holdout_idx
    }

    for fold, (train_idx, val_idx) in enumerate(cv_splits):

        split_data[f"fold_{fold + 1}_train_idx"] = train_idx
        split_data[f"fold_{fold + 1}_val_idx"] = val_idx

    np.savez_compressed(
        SPLITS_OUTPUT,
        **split_data
    )

    # ======================================
    # Optuna Optimization
    # ======================================

    print("\n========== OPTUNA OPTIMIZATION ==========")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=1,
            interval_steps=1
        )
    )

    study.optimize(
        lambda trial: objective_gemv_log(
            trial,
            dev_df,
            cv_splits
        ),
        n_trials=N_TRIALS
    )

    # ======================================
    # Extract Best Trial
    # ======================================

    best_trial = study.best_trial

    best_params = dict(best_trial.params)

    fold_rmses = best_trial.user_attrs["fold_rmses"]

    fold_best_rounds = best_trial.user_attrs[
        "fold_best_rounds"
    ]

    # Median of the best boosting rounds across 5 folds
    median_best_rounds = float(
        np.median(fold_best_rounds)
    )

    # ======================================
    # Adjust Final n_estimators
    # ======================================

    final_n_estimators = int(
        np.ceil(
            median_best_rounds *
            FINAL_ESTIMATOR_MULTIPLIER
        )
    )

    best_params["n_estimators"] = final_n_estimators

    # ======================================
    # Save Best Parameters
    # ======================================

    with open(PARAMS_OUTPUT, "w") as f:
        json.dump(best_params, f, indent=4)

    # Save CV diagnostics separately
    cv_summary = {
        "best_cv_rmse": float(best_trial.value),

        "fold_rmses": fold_rmses,

        "mean_cv_rmse": float(np.mean(fold_rmses)),

        "std_cv_rmse": float(np.std(fold_rmses)),

        "fold_best_rounds": fold_best_rounds,

        "median_cv_best_rounds": median_best_rounds,

        "final_estimator_multiplier": FINAL_ESTIMATOR_MULTIPLIER,

        "final_n_estimators": final_n_estimators,

        "n_splits": N_SPLITS,

        "n_trials": N_TRIALS
    }

    with open(CV_SUMMARY_OUTPUT, "w") as f:
        json.dump(cv_summary, f, indent=4)

    # ======================================
    # Print Results
    # ======================================

    print("\n========== OPTIMIZATION COMPLETE ==========")

    print(f"Best mean CV RMSE: {best_trial.value:.6f}")

    print("\nFold RMSEs:")

    for i, score in enumerate(fold_rmses):
        print(f"Fold {i + 1}: {score:.6f}")

    print("\nFold best boosting rounds:")

    for i, rounds in enumerate(fold_best_rounds):
        print(f"Fold {i + 1}: {rounds}")

    print(f"\nMedian best rounds: {median_best_rounds}")

    print(
        f"Final n_estimators "
        f"(median * {FINAL_ESTIMATOR_MULTIPLIER}): "
        f"{final_n_estimators}"
    )

    print(f"\nSaved parameters: {PARAMS_OUTPUT}")
    print(f"Saved CV summary: {CV_SUMMARY_OUTPUT}")
    print(f"Saved split indices: {SPLITS_OUTPUT}")

    print("\nThe 20% holdout dataset was NOT used for tuning.")
