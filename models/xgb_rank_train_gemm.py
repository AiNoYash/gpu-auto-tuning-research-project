import pandas as pd
import numpy as np
import xgboost as xgb
import json
from scipy.stats import spearmanr, ConstantInputWarning        
from sklearn.model_selection import GroupShuffleSplit
import warnings

warnings.filterwarnings("ignore", category=ConstantInputWarning)

def preprocess_data_gemm(raw_df):
    """Filters columns, adds derived features, and correctly penalizes invalid configurations."""
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
    df['TOTAL_THREADS'] = df['THREAD_X'] * df['THREAD_Y'] * df['THREAD_Z']
    df['GRID_SIZE'] = np.ceil(df['M'] / df['TILE_M']) * np.ceil(df['N'] / df['TILE_N'])
    df['WORK_PER_THREAD_M'] = (df['TILE_M'] + df['THREAD_Y'] - 1) // df['THREAD_Y']
    df['WORK_PER_THREAD_N'] = (df['TILE_N'] + df['THREAD_X'] - 1) // df['THREAD_X']
    df['SHMEM_FOOTPRINT'] = (df['TILE_M'] * df['TILE_K']) + (df['TILE_K'] * df['TILE_N'])
    df['UNROLL_ALIGNMENT'] = (df['TILE_K'] % df['UNROLL_FACTOR'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_M'] = (df['M'] % df['TILE_M'] == 0).astype(int)
    df['BOUNDARY_ALIGNMENT_N'] = (df['N'] % df['TILE_N'] == 0).astype(int)
    


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
    
    # 2. Setup Ranking Targets and Grouping
    df['qid'] = df.groupby(['M', 'N', 'K']).ngroup()
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
    with open('./hyperparams/xgb_rank_best_params_gemm.json', 'r') as f:
        best_params = json.load(f)
        
    # Extract the boosted rounds dynamically assigned during Optuna
    num_boost_round = best_params.pop("n_estimators", 3000)
    
    # Re-inject the fixed structural parameters
    best_params.update({
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "random_state": 42
    })
    
    # 6. Train the Model
    print("\nTraining final XGBoost model...")
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
    model_path = 'xgb_rank_model_gemm.json'
    bst.save_model(model_path)
    print(f"Model successfully saved to {model_path} for PSO inference.")