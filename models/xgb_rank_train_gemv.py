import pandas as pd
import numpy as np
import xgboost as xgb
import json
from scipy.stats import spearmanr, ConstantInputWarning        
from sklearn.model_selection import GroupShuffleSplit
import warnings

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
    df['GRID_SIZE'] = (df['M'] + df['ROWS_PER_BLOCK'] - 1) // df['ROWS_PER_BLOCK']
    df['SHMEM_ELEMENTS'] = df['TILE_K'] + (df['ROWS_PER_BLOCK'] * df['BLOCK_DIM_X'])
    df['WORK_PER_THREAD_K'] = (df['TILE_K'] + df['BLOCK_DIM_X'] - 1) // df['BLOCK_DIM_X']
    df['TILE_K_ALIGNED'] = (df['TILE_K'] % df['BLOCK_DIM_X'] == 0).astype(int)
    df['UNROLL_ALIGNMENT'] = (df['WORK_PER_THREAD_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['ROWS_PER_BLOCK'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_K'] = (df['K'] % df['TILE_K'] == 0).astype(int)
    
            
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


if __name__ == "__main__":
    # 1. Load and Preprocess Data
    raw_df = pd.read_csv('../datasets/t4/gemv/gemv_dataset.csv')
    df = preprocess_data_gemv(raw_df)
    
    features = [
        'M', 'K',
        'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR',
        'GRID_SIZE', 'SHMEM_ELEMENTS', 'WORK_PER_THREAD_K', 
        'TILE_K_ALIGNED', 'UNROLL_ALIGNMENT', 
        'BOUNDARY_ALIGNMENT_M', 'BOUNDARY_ALIGNMENT_K'
    ]
    
    # 2. Setup Ranking Targets and Grouping (GEMV only uses M and K)
    df['qid'] = df.groupby(['M', 'K']).ngroup()
    df['target'] = df.groupby('qid')['latency_ms'].rank(ascending=False, method='dense').astype(int)
    
    # CRITICAL: Sort by qid for XGBoost pairwise ranking
    df = df.sort_values('qid').reset_index(drop=True)
    
    # 3. Perform 80/20 Group-Aware Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['qid']))
    
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    print(f"Training on {len(train_data)} configurations across {train_data['qid'].nunique()} matrix sizes.")
    print(f"Testing on {len(test_data)} configurations across {test_data['qid'].nunique()} matrix sizes.")
    
    # 4. Create DMatrices
    dtrain = xgb.DMatrix(train_data[features], label=train_data['target'], qid=train_data['qid'])
    dtest = xgb.DMatrix(test_data[features], label=test_data['target'], qid=test_data['qid'])
    
    # 5. Load Hyperparameters
    with open('./hyperparams/xgb_rank_best_params_gemv.json', 'r') as f:
        best_params = json.load(f)
        
    # Extract the boosted rounds (Assuming you already manually handled the 1.1x trick in the JSON)
    num_boost_round = int(best_params.pop("n_estimators", 500))
    
    # Re-inject the fixed structural parameters (remembering ndcg_exp_gain: False!)
    best_params.update({
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "random_state": 42
    })
    
    # 6. Train the Model
    print("\nTraining final XGBoost ranking model for GEMV...")
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=50  # Print progress every 50 rounds
    )
    
    # 7. Evaluate on Test Set
    preds = bst.predict(dtest)
    test_spearman = calculate_grouped_spearman(test_data['target'].values, preds, test_data['qid'].values)
    print(f"\nFinal Test Spearman Correlation: {test_spearman:.4f}")
    
    # 8. Save Final Model for PSO
    model_path = 'xgb_rank_model_gemv.json'
    bst.save_model(model_path)
    print(f"Model successfully saved to {model_path} for PSO inference.")