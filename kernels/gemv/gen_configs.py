import math
import csv
from scipy.stats import qmc

param_space = {
    "M": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 8192, 14336],
    "K": [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192],
    "BLOCK_DIM_X": [32, 64, 128, 256, 512, 1024],
    "ROWS_PER_BLOCK": [1, 2, 4, 8, 16, 32],
    "TILE_K": [4, 8, 16, 32, 64, 128, 256],
    "UNROLL_FACTOR": [1, 2, 4, 8, 16, 32, 64]
}

def generate_discrete_lhs(space, num_samples):
    keys = list(space.keys())
    dimensions = len(keys)
    sampler = qmc.LatinHypercube(d=dimensions)
    
    configurations = []
    while len(configurations) < num_samples:
        raw_samples = sampler.random(n=num_samples)
        for point in raw_samples:
            config = {}
            for i, key in enumerate(keys):
                choices = space[key]
                idx = math.floor(point[i] * len(choices))
                config[key] = choices[idx]
            configurations.append(config)
            
            if len(configurations) == num_samples:
                return configurations
                
    return configurations

num_samples = 7500
configurations = generate_discrete_lhs(param_space, num_samples)

config_filename = "gemv_configurations.csv"
fieldnames = list(param_space.keys())

with open(config_filename, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(configurations)

print(f"Successfully generated {num_samples} configurations in '{config_filename}'.")