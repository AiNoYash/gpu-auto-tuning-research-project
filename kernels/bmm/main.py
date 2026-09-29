import subprocess
import random
import math
import os
import csv

# 1. Define discrete tuning space for Batched GEMM (BMM)
# All parameters restricted to clean powers-of-2 for Phase 1
param_space = {
    "BATCH_SIZE": [8, 16, 32, 64],
    "M": [64, 128, 256, 512, 1024],
    "N": [64, 128, 256, 512, 1024],
    "K": [64, 128, 256, 512],
    "TILE_M": [16, 32, 64, 128],
    "TILE_N": [16, 32, 64, 128],
    "TILE_K": [8, 16, 32],
    "THREAD_X": [8, 16, 32],
    "THREAD_Y": [4, 8, 16],
    "THREAD_Z": [1],
}

def generate_discrete_lhs(space, num_samples):
    """
    Generates a Latin Hypercube Sample for discrete categorical data.
    Ensures every value in a parameter's list is sampled as evenly as possible.
    """
    lhs_samples = {}
    for key, choices in space.items():
        repeats = math.ceil(num_samples / len(choices))
        extended_choices = (choices * repeats)[:num_samples]
        random.shuffle(extended_choices)
        lhs_samples[key] = extended_choices
    
    configurations = []
    for i in range(num_samples):
        config = {key: lhs_samples[key][i] for key in space.keys()}
        configurations.append(config)
        
    return configurations

def main():
    # Number of samples to collect
    num_samples = 50
    configurations = generate_discrete_lhs(param_space, num_samples)

    csv_filename = "bmm_tuning_dataset.csv"
    fieldnames = list(param_space.keys()) + ["latency_ms"]
    file_exists = os.path.isfile(csv_filename)

    print(f"Starting Batched GEMM (BMM) LHS Autotuning: {num_samples} configurations...")

    # 2. Iterate and compile/run
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for idx, config in enumerate(configurations):
            print(f"\nEvaluating Config {idx+1}/{num_samples}: {config}")
            
            # Inject compilation flags (compile-time tuning parameters)
            compile_cmd = [
                "nvcc", "-O3", "main.cu", "-o", "bmm_autotune",
                f"-DTILE_M={config['TILE_M']}", f"-DTILE_N={config['TILE_N']}", f"-DTILE_K={config['TILE_K']}",
                f"-DTHREAD_X={config['THREAD_X']}", f"-DTHREAD_Y={config['THREAD_Y']}", f"-DTHREAD_Z={config['THREAD_Z']}"
            ]
            
            try:
                # Compile the CUDA C++ code
                subprocess.run(compile_cmd, check=True, capture_output=True)
                
                # Pass M, N, K, and BATCH_SIZE at runtime
                run_cmd = [
                    "./bmm_autotune",
                    str(config['M']),
                    str(config['N']),
                    str(config['K']),
                    str(config['BATCH_SIZE'])
                ]
                result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
                
                exec_time_ms = float(result.stdout.strip())
                
                if exec_time_ms < 0:
                    print(f"-> Invalid Configuration: Exceeded hardware limits (threads/shared memory/launch failed). Skipped.")
                    continue

                # Merge execution time into config dictionary and save
                data_point = config.copy()
                data_point["latency_ms"] = exec_time_ms
                
                writer.writerow(data_point)
                csvfile.flush()
                
                print(f"-> Success: {exec_time_ms:.4f} ms")
                
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode('utf-8').strip() if e.stderr else str(e)
                print(f"-> Failed Compilation/Execution. Error: {err_msg}")
            except ValueError:
                print(f"-> Failed Parsing Output. Raw output: {result.stdout.strip()}")

if __name__ == "__main__":
    main()
