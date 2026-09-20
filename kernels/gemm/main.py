import subprocess
import random
import math
import os
import csv

# 1. Define the discrete tuning space for all 10 parameters
# Using powers of 2 / multiples of warp size to ensure meaningful hardware mapping
param_space = {
    "M": [512, 1024, 2048, 4096],
    "N": [512, 1024, 2048, 4096],
    "K": [512, 1024, 2048],
    "TILE_M": [32, 64, 128],
    "TILE_N": [32, 64, 128],
    "TILE_K": [8, 16, 32],
    "THREAD_X": [8, 16, 32],
    "THREAD_Y": [4, 8, 16],
    "THREAD_Z": [1], # Z is typically 1 for 2D tiling
    "UNROLL": [1, 2, 4, 8]
}

def generate_discrete_lhs(space, num_samples):
    """
    Generates a Latin Hypercube Sample for discrete categorical data.
    Ensures every value in a parameter's list is sampled as evenly as possible.
    """
    lhs_samples = {}
    for key, choices in space.items():
        # Repeat the choices enough times to cover the requested number of samples
        repeats = math.ceil(num_samples / len(choices))
        extended_choices = (choices * repeats)[:num_samples]
        # Shuffle independently to break correlations between dimensions
        random.shuffle(extended_choices)
        lhs_samples[key] = extended_choices
    
    # Restructure from dictionary of lists to list of dictionaries
    configurations = []
    for i in range(num_samples):
        config = {key: lhs_samples[key][i] for key in space.keys()}
        configurations.append(config)
        
    return configurations

# Generate 50 random samples across the hypercube
num_samples = 50
configurations = generate_discrete_lhs(param_space, num_samples)

csv_filename = "gemm_tuning_dataset.csv"
fieldnames = list(param_space.keys()) + ["latency_ms"]
file_exists = os.path.isfile(csv_filename)

print(f"Starting LHS Autotuning: {num_samples} configurations...")

# 2. Iterate and compile/run
# Open the CSV in append mode so data is saved immediately after every run
with open(csv_filename, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header if the file was just created
    if not file_exists:
        writer.writeheader()

    for idx, config in enumerate(configurations):
        print(f"\nEvaluating Config {idx+1}/{num_samples}: {config}")
        
        # Inject compilation flags
        compile_cmd = [
            "nvcc", "-O3", "main.cu", "-o", "gemm_autotune",
            f"-DTILE_M={config['TILE_M']}", f"-DTILE_N={config['TILE_N']}", f"-DTILE_K={config['TILE_K']}",
            f"-DTHREAD_X={config['THREAD_X']}", f"-DTHREAD_Y={config['THREAD_Y']}", f"-DTHREAD_Z={config['THREAD_Z']}",
            f"-DUNROLL_FACTOR={config['UNROLL']}"
        ]
        
        try:
            # Compile the C++ code
            subprocess.run(compile_cmd, check=True, capture_output=True)
            
            # 3. Run the compiled binary and pass the Matrix shape dynamically
            run_cmd = ["./gemm_autotune", str(config['M']), str(config['N']), str(config['K'])]
            result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
            
            # Parse the execution time output by main.cu
            exec_time_ms = float(result.stdout.strip())
            
            # Merge execution time into the config dictionary for CSV writing
            data_point = config.copy()
            data_point["latency_ms"] = exec_time_ms
            
            # Write directly to CSV
            writer.writerow(data_point)
            csvfile.flush() # Ensure it writes to disk immediately
            
            print(f"-> Success: {exec_time_ms:.4f} ms")
            
        except subprocess.CalledProcessError as e:
            # Catch invalid configurations (e.g., TILE_M not divisible by THREAD_Y, or excessive shared memory)
            print(f"-> Failed Compilation/Execution. Error: {e.stderr.decode('utf-8').strip()}")
        except ValueError:
            print(f"-> Failed Parsing Output. Raw output: {result.stdout.strip()}")