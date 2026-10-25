import subprocess
import os
import csv
import json

start_idx = 0        
num_to_evaluate = 500 

config_filename = "gemm_configurations.csv"
dataset_filename = "gemm_tuning_dataset_1.csv"

# Read the generated configurations
configurations = []
with open(config_filename, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        configurations.append(row)

chunk_to_evaluate = configurations[start_idx : start_idx + num_to_evaluate]

# Ensure we have the base fields from the input CSV, plus our output metrics
if len(configurations) > 0:
    base_fields = list(configurations[0].keys())
else:
    base_fields = []
    
extra_fields = ["latency_ms", "status", "mean_ms", "variance", "rel_variance", "iterations"]
fieldnames = base_fields + extra_fields

file_exists = os.path.isfile(dataset_filename)

print(f"Evaluating configurations {start_idx} to {start_idx + len(chunk_to_evaluate) - 1}...")

with open(dataset_filename, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()

    for i, config in enumerate(chunk_to_evaluate):
        global_idx = start_idx + i
        print(f"\nEvaluating Config {global_idx+1}/{len(configurations)}: {config}")
        
        compile_cmd = [
            "nvcc", "-O3", "main.cu", "-arch=sm_75", "-o", "gemm_autotune",
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
                data_point.update({"latency_ms": -1.0, "status": "timeout", "mean_ms": -1.0, "variance": -1.0, "rel_variance": -1.0, "iterations": 0})
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
                status = output_data.get("status", "unknown_error")
                error_msg = output_data.get("error_message", "Unknown error")
                print(f"-> Invalid Configuration ({status}): {error_msg}. Recording as -1.0.")
                
                data_point.update({"latency_ms": -1.0, "status": status, "mean_ms": -1.0, "variance": -1.0, "rel_variance": -1.0, "iterations": 0})
            
            writer.writerow(data_point)
            csvfile.flush()
            
        except subprocess.CalledProcessError as e:
            # Catches actual compilation errors from nvcc
            print(f"-> Failed Compilation.")
            
            data_point = config.copy()
            data_point.update({"latency_ms": -1.0, "status": "compilation_failed", "mean_ms": -1.0, "variance": -1.0, "rel_variance": -1.0, "iterations": 0})
            
            writer.writerow(data_point)
            csvfile.flush()