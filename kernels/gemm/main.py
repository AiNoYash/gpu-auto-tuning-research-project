import subprocess
import random
import math
import os
import csv
import json


param_space = {
    "M": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 14336],
    "N": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 14336],
    "K": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192],
    
    "TILE_M": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_N": [4, 8, 16, 32, 64, 96, 128, 192, 256],
    "TILE_K": [2, 4, 8, 16, 32, 64],
    
    "THREAD_X": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "THREAD_Y": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "THREAD_Z": [1], 
    "UNROLL_FACTOR": [1, 2, 4, 8, 16, 32, 64]
}

def generate_discrete_lhs(space, num_samples):
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

num_samples = 500
configurations = generate_discrete_lhs(param_space, num_samples)

csv_filename = "gemm_tuning_dataset.csv"

# Add the new JSON stats to the CSV headers
extra_fields = ["latency_ms", "status", "mean_ms", "variance", "rel_variance", "iterations"]
fieldnames = list(param_space.keys()) + extra_fields

file_exists = os.path.isfile(csv_filename)

print(f"Starting LHS Autotuning: {num_samples} configurations...")

with open(csv_filename, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()

    for idx, config in enumerate(configurations):
        print(f"\nEvaluating Config {idx+1}/{num_samples}: {config}")
        
        compile_cmd = [
                "nvcc", "-O3", "main.cu", "-o", "gemm_autotune",
                f"-DTILE_M={config['TILE_M']}", f"-DTILE_N={config['TILE_N']}", f"-DTILE_K={config['TILE_K']}",
                f"-DTHREAD_X={config['THREAD_X']}", f"-DTHREAD_Y={config['THREAD_Y']}", f"-DTHREAD_Z={config['THREAD_Z']}",
                f"-DUNROLL_FACTOR={config['UNROLL_FACTOR']}"
            ]

        try:
            # Compile the C++ code
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            
            # Run the compiled binary
            run_cmd = ["./gemm_autotune", str(config['M']), str(config['N']), str(config['K'])]
            
                        
            try:
                result = subprocess.run(run_cmd, check=True, capture_output=True, text=True, timeout=100.0)
            except subprocess.TimeoutExpired:
                print("-> Execution Timeout (Config took >10s). Recording as -1.0.")
                data_point = config.copy()
                data_point["latency_ms"] = -1.0
                data_point["status"] = "timeout"
                data_point["mean_ms"] = -1.0
                data_point["variance"] = -1.0
                data_point["rel_variance"] = -1.0
                data_point["iterations"] = 0
                writer.writerow(data_point)
                csvfile.flush()
                continue 
            
            # Parse the JSON execution output
            try:
                output_data = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                print(f"-> Failed Parsing JSON. Raw output: {result.stdout.strip()}")
                continue
            
            data_point = config.copy()
            
            # Handle the JSON response and map it to CSV columns
            if output_data.get("status") == "success":
                print(f"-> Success: {output_data['median_ms']:.4f} ms (Runs: {output_data['iterations']}, RelVar: {output_data['rel_variance']:.5f})")
                
                data_point["latency_ms"] = output_data["median_ms"]
                data_point["status"] = output_data["status"]
                data_point["mean_ms"] = output_data["mean_ms"]
                data_point["variance"] = output_data["variance"]
                data_point["rel_variance"] = output_data["rel_variance"]
                data_point["iterations"] = output_data["iterations"]
                
            else:
                # Catches 'invalid_config', 'cuda_oom', etc.
                status = output_data.get("status", "unknown_error")
                error_msg = output_data.get("error_message", "Unknown error")
                print(f"-> Invalid Configuration ({status}): {error_msg}. Recording as -1.0.")
                
                data_point["latency_ms"] = -1.0
                data_point["status"] = status
                data_point["mean_ms"] = -1.0
                data_point["variance"] = -1.0
                data_point["rel_variance"] = -1.0
                data_point["iterations"] = 0
            
            writer.writerow(data_point)
            csvfile.flush()
            
        except subprocess.CalledProcessError as e:
            # Catches actual compilation errors from nvcc
            print(f"-> Failed Compilation. Error: {e.stderr.strip()}")
            
            data_point = config.copy()
            data_point["latency_ms"] = -1.0
            data_point["status"] = "compilation_failed"
            data_point["mean_ms"] = -1.0
            data_point["variance"] = -1.0
            data_point["rel_variance"] = -1.0
            data_point["iterations"] = 0
            
            writer.writerow(data_point)
            csvfile.flush()