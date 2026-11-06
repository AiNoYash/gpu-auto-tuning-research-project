import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

def preprocess_data_gemv_log(raw_df):
    """Filters columns, adds derived features for GEMV, and applies log transformation."""
    keep_cols = [
        'M', 'K', 
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Penalize invalid configs (-1.0) with a massive latency penalty
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
    

    return df

if __name__ == "__main__":
    # 1. Load and Preprocess Data
    raw_df = pd.read_csv('../datasets/t4/gemv/gemv_dataset.csv')
    df = preprocess_data_gemv_log(raw_df)
    
    features = [
        'M', 'K',
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR',
        'GRID_SIZE', 'SHMEM_ELEMENTS', 'WORK_PER_THREAD_K', 
        'TILE_K_ALIGNED', 'UNROLL_ALIGNMENT', 
        'BOUNDARY_ALIGNMENT_M', 'BOUNDARY_ALIGNMENT_K'
    ]
    
    # 2. Setup Grouping to prevent data leakage across matrix sizes (M, K)
    df['qid'] = df.groupby(['M', 'K']).ngroup()
    
    # 3. Perform 80/20 Group-Aware Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    print(f"Training on {len(train_data)} configurations across {train_data['qid'].nunique()} matrix sizes.")
    print(f"Testing on {len(test_data)} configurations across {test_data['qid'].nunique()} matrix sizes.")
    
    # 4. Create DMatrices 
    # NOTE: No 'qid' is passed here because this is standard regression!
    dtrain = xgb.DMatrix(train_data[features], label=train_data['log_latency'])
    dtest = xgb.DMatrix(test_data[features], label=test_data['log_latency'])
    
    # 5. Load Hyperparameters
    with open('./hyperparams/xgb_logexe_best_params_gemv.json', 'r') as f:
        best_params = json.load(f)
        
    # Extract the boosted rounds dynamically assigned during Optuna
    num_boost_round = int(best_params.pop("n_estimators", 500))
    
    # Re-inject the fixed structural parameters for log-regression
    best_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42
    })
    
    # 6. Train the Model
    print("\nTraining final XGBoost Log-Regression model for GEMV...")
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=50  # Print progress every 50 rounds
    )
    
    # 7. Evaluate on Test Set
    preds_log = bst.predict(dtest)
    
    # Calculate RMSE in the log-space (this is the true metric of landscape learning)
    test_rmse_log = np.sqrt(mean_squared_error(test_data['log_latency'], preds_log))
    print(f"\nFinal Test RMSE (Log Space): {test_rmse_log:.4f}")
    
    # 8. Save Final Model for PSO
    model_path = 'xgb_logexe_model_gemv.json'
    bst.save_model(model_path)
    print(f"Model successfully saved to {model_path} for PSO inference.")