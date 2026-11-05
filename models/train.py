import pandas as pd
import xgboost as xgb
import json

# ==========================================
# 1. Preprocessing Function (Identical to Tuning)
# ==========================================
def preprocess_and_prepare_data(raw_df):
    """Filters columns, penalizes invalid configs, and builds ranking targets."""
    keep_cols = [
        'M', 'N', 'K', 
        'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR', 
        'latency_ms'
    ]
    df = raw_df[keep_cols].copy()
    
    # Convert invalid configs (-1.0) to a massive latency penalty
    df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e8
    
    # Create the QID 
    df['qid'] = df.groupby(['M', 'N', 'K']).ngroup()

    # Create integer ranks for the pairwise objective
    df['target'] = df.groupby('qid')['latency_ms'].rank(ascending=False, method='dense').astype(int)
    
    # CRITICAL: Sort by qid for XGBoost
    df = df.sort_values('qid').reset_index(drop=True)
    
    return df

# ==========================================
# 2. Main Training Execution
# ==========================================
if __name__ == "__main__":
    # 1. Load the Data
    print("Loading and preprocessing dataset...")
    raw_df = pd.read_csv('../datasets/t4/gemm/gemm_dataset.csv')
    df = preprocess_and_prepare_data(raw_df)
    
    features = ['TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR']
    
    # Train on 100% of the data
    dtrain = xgb.DMatrix(df[features], label=df['target'], qid=df['qid'])
    
    # 2. Load the Optimized Hyperparameters
    print("Loading hyperparameters...")
    with open('xgboost_best_params.json', 'r') as f:
        best_params = json.load(f)
        
    # Extract n_estimators (used as num_boost_round in the native API)
    # Default to 1000 just in case it was missing from the JSON
    num_boost_round = best_params.pop('n_estimators', 1000)
    
    # Ensure static required parameters are present
    best_params['objective'] = 'rank:pairwise'
    best_params['tree_method'] = 'hist'
    
    print(f"Training final model with {num_boost_round} trees...")
    
    # 3. Train the Final Model
    # Notice we do not use early stopping or a validation set here
    bst = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round
    )
    
    # 4. Save the Model for PSO Inference
    model_filename = "xgboost_pso_ranker.json"
    bst.save_model(model_filename)
    
    print(f"Success! Model saved to {model_filename}")