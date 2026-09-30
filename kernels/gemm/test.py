import subprocess
import json

# ---- Fixed parameters (edit these if your kernel needs different values) ----
FIXED = {
    "M": 4096, "N": 4096, "K": 4096,       # large enough to get measurable time
    "TILE_M": 64, "TILE_N": 64, "TILE_K": 64,
    "THREAD_X": 16, "THREAD_Y": 16, "THREAD_Z": 1,
}

# ---- The only thing that changes ----
UNROLL_FACTORS = [1, 2, 4, 8, 16, 32, 64]

BINARY = "gemm_unroll_test"
TIMEOUT_S = 100.0

print("Fixed config:", FIXED)
print(f"Sweeping UNROLL_FACTOR over {UNROLL_FACTORS}\n")

results = []

for uf in UNROLL_FACTORS:
    print(f"--- UNROLL_FACTOR = {uf} ---")

    compile_cmd = [
        "nvcc", "-O3", "main.cu", "-o", BINARY,
        f"-DTILE_M={FIXED['TILE_M']}", f"-DTILE_N={FIXED['TILE_N']}", f"-DTILE_K={FIXED['TILE_K']}",
        f"-DTHREAD_X={FIXED['THREAD_X']}", f"-DTHREAD_Y={FIXED['THREAD_Y']}", f"-DTHREAD_Z={FIXED['THREAD_Z']}",
        f"-DUNROLL_FACTOR={uf}",
    ]

    try:
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  compile failed: {e.stderr.strip()}\n")
        results.append((uf, None))
        continue

    run_cmd = [f"./{BINARY}", str(FIXED["M"]), str(FIXED["N"]), str(FIXED["K"])]
    try:
        out = subprocess.run(run_cmd, check=True, capture_output=True, text=True, timeout=TIMEOUT_S)
        data = json.loads(out.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"  timeout (> {TIMEOUT_S}s)\n")
        results.append((uf, None))
        continue
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"  run/parse failed: {e}\n")
        results.append((uf, None))
        continue

    if data.get("status") != "success":
        print(f"  {data.get('status')}: {data.get('error_message', 'unknown error')}\n")
        results.append((uf, None))
        continue

    print(f"  median={data['median_ms']:.4f} ms  mean={data['mean_ms']:.4f} ms  "
          f"rel_var={data['rel_variance']:.5f}  iters={data['iterations']}\n")
    results.append((uf, data["median_ms"]))

# ---- Summary ----
valid = [(uf, t) for uf, t in results if t is not None]
baseline = next((t for uf, t in valid if uf == 1), None)

print("=" * 44)
print(f"{'UNROLL':>8} {'median (ms)':>14} {'speedup vs 1':>16}")
for uf, t in results:
    if t is None:
        print(f"{uf:>8} {'failed':>14} {'-':>16}")
    else:
        sp = f"{baseline / t:.3f}x" if baseline else "-"
        print(f"{uf:>8} {t:>14.4f} {sp:>16}")

if valid:
    best_uf, best_t = min(valid, key=lambda x: x[1])
    print(f"\nBest: UNROLL_FACTOR={best_uf} ({best_t:.4f} ms)")