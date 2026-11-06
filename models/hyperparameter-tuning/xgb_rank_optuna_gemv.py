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

def preprocess_data_gemv(raw_df):
    """Filters columns, adds derived features for GEMV, and penalizes invalid configs."""
    keep_cols = [
        'M', 'K', 
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Penalize invalid configs (-1.0) with a massive latency penalty
    df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e8
    
    # ==========================================
    # Derived Features
    # ==========================================
    # 1. Grid Size (Total thread blocks launched)
    df['GRID_SIZE'] = (df['M'] + df['ROWS_PER_BLOCK'] - 1) // df['ROWS_PER_BLOCK']
    
    # 2. Shared Memory Elements (Proxy for occupancy limits)
    df['SHMEM_ELEMENTS'] = df['TILE_K'] + (df['ROWS_PER_BLOCK'] * df['BLOCK_DIM_X'])
    
    # 3. Work per Thread in K (Proxy for inner loop iterations / register pressure)
    df['WORK_PER_THREAD_K'] = (df['TILE_K'] + df['BLOCK_DIM_X'] - 1) // df['BLOCK_DIM_X']
    
    # 4. Perfect Tile Loading/Compute Balance (Avoids warp divergence)
    df['TILE_K_ALIGNED'] = (df['TILE_K'] % df['BLOCK_DIM_X'] == 0).astype(int)
    
    # 5. Perfect Unroll Alignment (Avoids compiler tail loops)
    df['UNROLL_ALIGNMENT'] = (df['WORK_PER_THREAD_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    
    # 6. Global M-Boundary Alignment (Avoids divergence on global_row < M)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['ROWS_PER_BLOCK'] == 0).astype(int)
    
    # 7. Global K-Boundary Alignment (Avoids divergence on gk < K)
    df['BOUNDARY_ALIGNMENT_K'] = (df['K'] % df['TILE_K'] == 0).astype(int)
    
    # Force 32-bit types to prevent XGBoost DMatrix casting bottlenecks
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
            
    return df

def calculate_grouped_spearman(y_true, y_pred, qids):
    """Calculates the average Spearman correlation across all qid groups."""
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred, 'qid': qids})
    spearman_scores = []
    
    for _, group in df_eval.groupby('qid'):
        if len(group) > 1:
            preds = group['pred'].values + np.random.normal(0, 1e-6, len(group))
            rho, _ = spearmanr(group['true'].values, preds)
            if not np.isnan(rho):
                spearman_scores.append(rho)
                
    return np.mean(spearman_scores) if spearman_scores else 0.0

def objective_gemv(trial, df):
    # Features and Target mapping
    features = [
        'M', 'K',
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR',
        'GRID_SIZE', 'SHMEM_ELEMENTS', 'WORK_PER_THREAD_K', 
        'TILE_K_ALIGNED', 'UNROLL_ALIGNMENT', 
        'BOUNDARY_ALIGNMENT_M', 'BOUNDARY_ALIGNMENT_K'
    ]
    
    # Group by M and K for GEMV
    df['qid'] = df.groupby(['M', 'K']).ngroup()

    # Rank latencies within each workload size (qid)
    df['target'] = df.groupby('qid')['latency_ms'].rank(ascending=False, method='dense').astype(int)
    
    # Sort by qid for XGBoost ranking
    df = df.sort_values('qid').reset_index(drop=True)
    
    # Group-aware train/validation split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    dtrain = xgb.DMatrix(train_data[features], label=train_data['target'], qid=train_data['qid'])
    dval = xgb.DMatrix(val_data[features], label=val_data['target'], qid=val_data['qid'])
    
    param = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "ndcg_exp_gain": False,
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
    
    pruning_callback = XGBoostPruningCallback(trial, "val-ndcg")
    
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose_eval=False
    )
    
    best_iter = getattr(bst, 'best_iteration', 500)
    trial.set_user_attr("best_iteration", best_iter)
    
    preds = bst.predict(dval)
    spearman_score = calculate_grouped_spearman(val_data['target'].values, preds, val_data['qid'].values)
    
    return spearman_score

if __name__ == "__main__":
    raw_df = pd.read_csv('../../datasets/t4/gemv/gemv_dataset.csv')
    
    df = preprocess_data_gemv(raw_df)
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    study.optimize(lambda trial: objective_gemv(trial, df), n_trials=300)
    
    best_params = study.best_trial.params
    best_iteration = study.best_trial.user_attrs.get("best_iteration", 500)
    best_params["n_estimators"] = int(best_iteration)
    
    with open('xgb_rank_best_params_gemv.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Saved optimized parameters to xgboost_gemv_best_params.json")