import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

def preprocess_data_gemm(raw_df):
    """Filters columns, adds derived features, and applies log transformation."""
    keep_cols = [
        'M', 'N', 'K', 
        'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Convert invalid configs (-1.0) to a massive latency penalty
    df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e8
    
    # Transform target to log(latency)
    df['log_latency'] = np.log(df['latency_ms'])
    
    # ==========================================
    # Derived Features
    # ==========================================
    df['TOTAL_THREADS'] = df['THREAD_X'] * df['THREAD_Y'] * df['THREAD_Z']
    df['GRID_SIZE'] = np.ceil(df['M'] / df['TILE_M']) * np.ceil(df['N'] / df['TILE_N'])
    df['WORK_PER_THREAD_M'] = (df['TILE_M'] + df['THREAD_Y'] - 1) // df['THREAD_Y']
    df['WORK_PER_THREAD_N'] = (df['TILE_N'] + df['THREAD_X'] - 1) // df['THREAD_X']
    df['SHMEM_FOOTPRINT'] = (df['TILE_M'] * df['TILE_K']) + (df['TILE_K'] * df['TILE_N'])
    df['UNROLL_ALIGNMENT'] = (df['TILE_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['TILE_M'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_N'] = (df['N'] % df['TILE_N'] == 0).astype(int)
    
    return df

if __name__ == "__main__":
    # 1. Load and Preprocess Data
    raw_df = pd.read_csv('../datasets/t4/gemm/gemm_dataset.csv')
    df = preprocess_data_gemm(raw_df)
    
    features = [
        "M", "N", "K",
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
    
    # 2. Setup Grouping to prevent data leakage across matrix sizes
    df['qid'] = df.groupby(['M', 'N', 'K']).ngroup()
    
    # 3. Perform 80/20 Group-Aware Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    print(f"Training on {len(train_data)} configurations across {train_data['qid'].nunique()} matrix sizes.")
    print(f"Testing on {len(test_data)} configurations across {test_data['qid'].nunique()} matrix sizes.")
    
    # 4. Create DMatrices 
    dtrain = xgb.DMatrix(train_data[features], label=train_data['log_latency'])
    dtest = xgb.DMatrix(test_data[features], label=test_data['log_latency'])
    
    # 5. Load Hyperparameters
    # Adjust path if your JSON is in a different directory
    with open('./hyperparams/xgb_logexe_best_params_gemm.json', 'r') as f:
        best_params = json.load(f)
        
    # Extract the boosted rounds dynamically assigned during Optuna
    num_boost_round = best_params.pop("n_estimators", 500)
    
    # Re-inject the fixed structural parameters for log-regression
    best_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42
    })
    
    # 6. Train the Model
    print("\nTraining final XGBoost Log-Regression model...")
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=50  # Print progress every 50 rounds
    )
    
    # 7. Evaluate on Test Set
    preds_log = bst.predict(dtest)
    
    # Calculate RMSE in the log-space
    test_rmse_log = np.sqrt(mean_squared_error(test_data['log_latency'], preds_log))
    print(f"\nFinal Test RMSE (Log Space): {test_rmse_log:.4f}")
    
    # (Optional) Calculate RMSE in standard millisecond space to see real-world error
    preds_ms = np.exp(preds_log)
    # We filter out the invalid configs (1e8) for the MS calculation so they don't skew the human-readable error
    valid_mask = test_data['latency_ms'] < 1e7 
    if valid_mask.sum() > 0:
        test_rmse_ms = np.sqrt(mean_squared_error(test_data['latency_ms'][valid_mask], preds_ms[valid_mask]))
        print(f"Final Test RMSE (Real Milliseconds, valid configs only): {test_rmse_ms:.4f} ms")
    
    # 8. Save Final Model for PSO
    model_path = 'xgb_logexe_model_gemm.json'
    bst.save_model(model_path)
    print(f"Model successfully saved to {model_path} for PSO inference.")