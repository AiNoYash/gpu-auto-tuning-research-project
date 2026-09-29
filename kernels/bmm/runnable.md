# Batched GEMM (BMM) Auto-Tuning

### Colab Execution Commands

1. **Navigate to the BMM kernel directory**:
   ```bash
   cd kernels/bmm/
   ```

2. **Run Full Latin Hypercube Auto-Tuning**:
   ```bash
   python main.py
   ```
   * Outputs dataset to `bmm_tuning_dataset.csv`.

3. **Manual Single Compilation & Run (Sanity Check)**:
   ```bash
   # Compile with specific tile and thread parameters
   nvcc -O3 main.cu -o bmm_autotune \
     -DTILE_M=64 -DTILE_N=64 -DTILE_K=16 \
     -DTHREAD_X=16 -DTHREAD_Y=16 -DTHREAD_Z=1

   # Run with M=512, N=512, K=512, BATCH_SIZE=32
   ./bmm_autotune 512 512 512 32
   ```
