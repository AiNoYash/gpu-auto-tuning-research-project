import numpy as np
import pandas as pd
import xgboost as xgb
import pyswarms as ps

# Load the saved surrogate model
model = xgb.XGBRegressor()
model.load_model("../models/xgb_gemm.json")

# Fixed variables
M_val, N_val, K_val = 1024, 1024, 512

# Search space definition
search_space = {
    "TILE_M": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_N": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_K": [2, 4, 8, 16, 32, 64],
    "THREAD_X": [2, 4, 8, 16, 32, 64, 128, 256, 512],
    "THREAD_Y": [2, 4, 8, 16, 32, 64, 128, 256, 512],
    "UNROLL_FACTOR": [1, 2, 4, 8, 16, 32, 64]
}

# Define boundaries for the 6 continuous index dimensions
# Min index is 0, Max index is the length of each list minus 1
min_bounds = np.zeros(6)
max_bounds = np.array([len(v) - 1 for v in search_space.values()])
bounds = (min_bounds, max_bounds)

# NOTE: Replace this list with the exact output of `X.columns.tolist()` from your training script
feature_columns = ['M', 'N', 'K', 'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR']

def xgboost_fitness(swarm_positions):
    """
    Evaluates the entire swarm in one batch.
    swarm_positions shape: (n_particles, 6)
    """
    # 1. Round to nearest integer to get discrete indices
    indices = np.round(swarm_positions).astype(int)
    
    # 2. Map indices back to actual hardware values for the whole batch
    batch_features = []
    for p in indices:
        row = [
            M_val, N_val, K_val,
            search_space["TILE_M"][p[0]],
            search_space["TILE_N"][p[1]],
            search_space["TILE_K"][p[2]],
            search_space["THREAD_X"][p[3]],
            search_space["THREAD_Y"][p[4]],
            1, # THREAD_Z
            search_space["UNROLL_FACTOR"][p[5]]
        ]
        batch_features.append(row)
    
    # 3. Create a DataFrame so XGBoost recognizes the feature names
    df_batch = pd.DataFrame(batch_features, columns=feature_columns)
    
    # 4. Predict log(latency) for the whole swarm at once
    predictions = model.predict(df_batch)
    return predictions

# Initialize PySwarms optimizer
# c1: cognitive parameter, c2: social parameter, w: inertia
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=6, options=options, bounds=bounds)

# Run the optimization
best_cost, best_pos = optimizer.optimize(xgboost_fitness, iters=50)

# Decode the final best configuration
best_idx = np.round(best_pos).astype(int)
best_config = {
    "TILE_M": search_space["TILE_M"][best_idx[0]],
    "TILE_N": search_space["TILE_N"][best_idx[1]],
    "TILE_K": search_space["TILE_K"][best_idx[2]],
    "THREAD_X": search_space["THREAD_X"][best_idx[3]],
    "THREAD_Y": search_space["THREAD_Y"][best_idx[4]],
    "UNROLL_FACTOR": search_space["UNROLL_FACTOR"][best_idx[5]],
}

print(f"\nBest predicted log(latency): {best_cost}")
print(f"Optimal configuration: {best_config}")