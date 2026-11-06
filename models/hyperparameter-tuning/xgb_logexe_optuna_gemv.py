import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna_integration.xgboost import XGBoostPruningCallback  
from sklearn.model_selection import GroupShuffleSplit
import json

def preprocess_data_gemv_log(raw_df):
    """Filters columns, adds derived features for GEMV, and applies log transformation."""
    keep_cols = [
        'M', 'K', 
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Convert invalid configs (-1.0) to a massive latency penalty
    df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e8
    
    # Transform target to log(latency)
    df['log_latency'] = np.log(df['latency_ms'])
    
    # ==========================================
    # Derived Features for GEMV
    # ==========================================
    df['GRID_SIZE'] = (df['M'] + df['ROWS_PER_BLOCK'] - 1) // df['ROWS_PER_BLOCK']
    df['SHMEM_ELEMENTS'] = df['TILE_K'] + (df['ROWS_PER_BLOCK'] * df['BLOCK_DIM_X'])
    df['WORK_PER_THREAD_K'] = (df['TILE_K'] + df['BLOCK_DIM_X'] - 1) // df['BLOCK_DIM_X']
    df['TILE_K_ALIGNED'] = (df['TILE_K'] % df['BLOCK_DIM_X'] == 0).astype(int)
    df['UNROLL_ALIGNMENT'] = (df['WORK_PER_THREAD_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['ROWS_PER_BLOCK'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_K'] = (df['K'] % df['TILE_K'] == 0).astype(int)
    
    # Force 32-bit types to prevent XGBoost DMatrix casting bottlenecks
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
            
    return df


def objective_gemv_log(trial, df):
    features = [
        "M", "K",
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR',
        'GRID_SIZE', 'SHMEM_ELEMENTS', 'WORK_PER_THREAD_K', 
        'TILE_K_ALIGNED', 'UNROLL_ALIGNMENT', 
        'BOUNDARY_ALIGNMENT_M', 'BOUNDARY_ALIGNMENT_K'
    ]    
    
    # Create the qid to group by matrix dimensions (prevent data leakage)
    df['qid'] = df.groupby(['M', 'K']).ngroup()
    
    # Group-aware train/validation split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    # Create XGBoost DMatrices (Target is log_latency)
    dtrain = xgb.DMatrix(train_data[features], label=train_data['log_latency'])
    dval = xgb.DMatrix(val_data[features], label=val_data['log_latency'])
    
    # Define Hyperparameter Search Space
    param = {
        "objective": "reg:squarederror", 
        "eval_metric": "rmse",              
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "tree_method": "hist"
    }
    
    # Setup Pruning Callback tied to the validation RMSE score
    pruning_callback = XGBoostPruningCallback(trial, "val-rmse")
    
    # Train Model
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose_eval=False
    )
    
    # Log best iteration for final retraining
    best_iter = getattr(bst, 'best_iteration', 500)
    trial.set_user_attr("best_iteration", best_iter)
    
    # Return the best RMSE on the validation set
    best_rmse = bst.best_score
    return best_rmse

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    raw_df = pd.read_csv('../../datasets/t4/gemv/gemv_dataset.csv')
    
    df = preprocess_data_gemv_log(raw_df)
    
    # Direction is MINIMIZE for RMSE
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    study.optimize(lambda trial: objective_gemv_log(trial, df), n_trials=300)
    
    best_params = study.best_trial.params
    best_iteration = study.best_trial.user_attrs.get("best_iteration", 500)
    
    # 1.1x trick for full dataset retraining
    best_params["n_estimators"] = int(best_iteration)
    
    with open('xgb_logexe_best_params_gemv.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Saved optimized parameters to xgb_logexe_best_params_gemv.json")