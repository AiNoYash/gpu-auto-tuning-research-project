import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna_integration.xgboost import XGBoostPruningCallback  
from scipy.stats import spearmanr, ConstantInputWarning        
from sklearn.model_selection import GroupShuffleSplit
import warnings
import json

# Suppress scipy spearman warnings for very small groups
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def preprocess_data_gemm(raw_df):
    """Filters columns, adds derived features, and correctly penalizes invalid configurations."""
    # Keep only the essential columns for grouping, training, and target
    keep_cols = [
        'M', 'N', 'K', 
        'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Convert invalid configs (-1.0) to a massive latency penalty
    df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e8
    
    # ==========================================
    # Derived Features
    # ==========================================
    
    # 1. Total Threads per Block (Occupancy constraint proxy)
    df['TOTAL_THREADS'] = df['THREAD_X'] * df['THREAD_Y'] * df['THREAD_Z']
    
    # 2. Grid Size (Total thread blocks launched)
    df['GRID_SIZE'] = np.ceil(df['M'] / df['TILE_M']) * np.ceil(df['N'] / df['TILE_N'])
    
    # 3. Work per Thread (Thread coarsening/register pressure proxy)
    df['WORK_PER_THREAD_M'] = (df['TILE_M'] + df['THREAD_Y'] - 1) // df['THREAD_Y']
    df['WORK_PER_THREAD_N'] = (df['TILE_N'] + df['THREAD_X'] - 1) // df['THREAD_X']

    # 4. Shared Memory Footprint Proxy (Elements loaded into L1/Shared Memory)
    df['SHMEM_FOOTPRINT'] = (df['TILE_M'] * df['TILE_K']) + (df['TILE_K'] * df['TILE_N'])
    
    # 5. Unroll Step Alignment (1 if no tail loop needed, 0 otherwise)
    df['UNROLL_ALIGNMENT'] = (df['TILE_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    
    # 6. Global Boundary Alignment (1 if no boundary checks needed, 0 otherwise)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['TILE_M'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_N'] = (df['N'] % df['TILE_N'] == 0).astype(int)
    
    return df



# ==========================================
# 1. Custom Grouped Spearman Metric for Optuna
# ==========================================
def calculate_grouped_spearman(y_true, y_pred, qids):
    """Calculates the average Spearman correlation across all qid groups."""
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred, 'qid': qids})
    spearman_scores = []
    
    for _, group in df_eval.groupby('qid'):
        if len(group) > 1: # Spearman requires at least 2 items
            # Add small noise to predictions to break exact ties which confuse scipy
            preds = group['pred'].values + np.random.normal(0, 1e-6, len(group))
            rho, _ = spearmanr(group['true'].values, preds)
            if not np.isnan(rho):
                spearman_scores.append(rho)
                
    return np.mean(spearman_scores) if spearman_scores else 0.0

# ==========================================
# 2. Optuna Objective Function
# ==========================================
def objective(trial, df):
    
    # Features and Target mapping
    features = [
        'TILE_M', 'TILE_N', 'TILE_K', 
        'THREAD_X', 'THREAD_Y', 'THREAD_Z', 
        'UNROLL_FACTOR',
        'TOTAL_THREADS', 
        'GRID_SIZE', 
        'WORK_PER_THREAD_M', 'WORK_PER_THREAD_N', 
        'SHMEM_FOOTPRINT', 
        'UNROLL_ALIGNMENT', 
        'BOUNDARY_ALIGNMENT_M', 'BOUNDARY_ALIGNMENT_N'
    ]    
    # Create the qid first so we can group by it
    df['qid'] = df.groupby(['M', 'N', 'K']).ngroup()

    # Rank latencies within each workload size (qid).
    # ascending=False means the highest latency gets rank 1, and the lowest latency gets the highest integer rank.
    df['target'] = df.groupby('qid')['latency_ms'].rank(ascending=False, method='dense').astype(int)
    
    # CRITICAL: XGBoost requires data to be sorted by qid
    df = df.sort_values('qid').reset_index(drop=True)
    
    # Group-aware train/validation split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    # Create XGBoost DMatrices
    dtrain = xgb.DMatrix(train_data[features], label=train_data['target'], qid=train_data['qid'])
    dval = xgb.DMatrix(val_data[features], label=val_data['target'], qid=val_data['qid'])
    
    # Define Hyperparameter Search Space
    param = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg", # Fast built-in metric for early stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "tree_method": "hist" # Highly recommended for speed
    }
    
    # Setup Pruning Callback tied to the validation NDCG score
    pruning_callback = XGBoostPruningCallback(trial, "val-ndcg")
    
    # Train Model
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose_eval=False
    )
    
    best_iter = getattr(bst, 'best_iteration', 1000)
    trial.set_user_attr("best_iteration", best_iter)
    
    # Generate predictions on the validation set for the final Optuna score
    preds = bst.predict(dval)
    
    # Calculate Custom Grouped Spearman
    spearman_score = calculate_grouped_spearman(val_data['target'].values, preds, val_data['qid'].values)
    
    return spearman_score

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":

    raw_df = pd.read_csv('../../datasets/t4/gemm/gemm_dataset.csv')
    
    # 1. Run Preprocessing
    df = preprocess_data_gemm(raw_df)
    
    # 2. Run Optimization
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    study.optimize(lambda trial: objective(trial, df), n_trials=300)
    
    # Extract Best Parameters
    best_params = study.best_trial.params
    
    # Safely retrieve the best iteration, default to 1000 if missing
    best_iteration = study.best_trial.user_attrs.get("best_iteration", 1000)
    
    # Apply the 1.1x n_estimators trick for full dataset retraining
    best_params["n_estimators"] = int(best_iteration * 1.1)
    
    # Save to JSON
    with open('xgb_rank_best_params_gemm.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Saved optimized parameters to xgboost_best_params.json")