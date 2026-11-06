import pandas as pd
import os

gpu = "t4"
kernel = "bmm"

key_cols_map = {
    "gemm": ['M', 'N', 'K', 'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR'],
    "bmm": ["BATCH_SIZE", 'M', 'N', 'K', 'TILE_M', 'TILE_N', 'TILE_K', 'THREAD_X', 'THREAD_Y', 'THREAD_Z', 'UNROLL_FACTOR'],
    "gemv":  ['M', 'K', 'BLOCK_DIM_X', 'ROWS_PER_BLOCK', 'TILE_K', 'UNROLL_FACTOR']
}

base_path = "../datasets/" + gpu + "/" + kernel + "/"
csv_files = [f"{base_path}{kernel}_tuning_dataset_{i}.csv" for i in range(1, 16)]
csv_files.append(f"{base_path}{kernel}_tuning_dataset-3-new.csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_6(2).csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_7(1).csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_8(1).csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_9(1).csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_10.1.csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_10.2.csv")
csv_files.append(f"{base_path}{kernel}_tuning_dataset_10.3.csv")

key_cols = key_cols_map[kernel]

dfs = []
for file in csv_files:
    if os.path.exists(file):
        dfs.append(pd.read_csv(file))
    else:
        print(f"Warning: {file} not found.")

if not dfs:
    print("Error: No CSV files found. Check your relative paths.")
    exit()

df = pd.concat(dfs, ignore_index=True)

# Conflict resolution: sort by iterations (asc), then rel_variance (asc)
df = df.sort_values(by=['iterations', 'rel_variance'], ascending=[True, True])

# Drop duplicates based on the configuration tuple, keeping the first row
df_dedup = df.drop_duplicates(subset=key_cols, keep='first')

# Output combined dataset
df_dedup.to_csv(f"combined_{kernel}_dataset.csv", index=False)
print(f"Successfully saved combined_{kernel}_dataset.csv")

# Output statistics document
with open(f"combined_{kernel}_statistics.txt", "w") as f:
    f.write(f"=== {kernel.upper()} Dataset Statistics ===\n\n")
    f.write(f"Total Rows (Pre-deduplication): {len(df)}\n")
    f.write(f"Total Rows (Post-deduplication): {len(df_dedup)}\n")
    f.write(f"Duplicates Removed: {len(df) - len(df_dedup)}\n\n")
    
    f.write("--- Status Distribution ---\n")
    f.write(df_dedup['status'].value_counts().to_string() + "\n\n")
    
    f.write("--- Comprehensive Column Statistics (Numerical) ---\n")
    # Get standard describe metrics (count, mean, std, min, 25%, 50%, 75%, max)
    desc = df_dedup.describe()
    
    # Add variance manually since describe() only gives standard deviation
    numeric_cols = df_dedup.select_dtypes(include='number')
    desc.loc['variance'] = numeric_cols.var()
    
    # Write stats per column for readability
    for col in desc.columns:
        f.write(f"\n[{col}] Metrics:\n")
        f.write(desc[col].to_string() + "\n")
    
    success_df = df_dedup[df_dedup['status'] == 'success']
    if not success_df.empty and 'latency_ms' in success_df.columns:
        f.write("\n--- Latency (ms) Metrics (Success Only) ---\n")
        succ_desc = success_df['latency_ms'].describe()
        succ_desc['variance'] = success_df['latency_ms'].var()
        f.write(succ_desc.to_string() + "\n\n")
        
    shape_cols = [c for c in ['M', 'N', 'K'] if c in df_dedup.columns]
    if len(shape_cols) == 3:
        f.write("--- Top 10 Most Common Matrix Shapes (M, N, K) ---\n")
        f.write(df_dedup[['M', 'N', 'K']].value_counts().head(10).to_string() + "\n")

print(f"Successfully saved combined_{kernel}_statistics.txt")