# CUDA Accelerated Softmax

A progressive GPU-optimization project — from a naive kernel to a warp-shuffle-reduced, vectorized implementation, benchmarked against PyTorch. Softmax is the operation at the core of every transformer's attention and every classifier's output layer; making it *fast and numerically stable* on a GPU touches nearly every important CUDA concept: the memory hierarchy, parallel reductions, warp-level primitives, vectorized memory access, and kernel fusion.

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

Subtracting the row max before exponentiating is the **numerical-stability trick** — without it, large inputs overflow `exp()` to infinity. Every kernel here implements it. Each row needs two reductions (a max, then a sum) plus an elementwise divide, so the whole project is really an exercise in optimizing **parallel reductions** on a GPU.

## Benchmark results

Fill this in after running on your GPU (`make run` + the PyTorch script). Illustrative targets on a mid-range Ampere card, shape `4096 x 4096`:

| Kernel           | Relative Speed | Key Technique              |
|------------------|----------------|----------------------------|
| Naive            | 1x (baseline)  | Serial reduction           |
| Shared Memory    | ~3–5x          | Tree reduction             |
| Warp Shuffle     | ~6–9x          | Register-level reduction   |
| Vectorized       | ~8–12x         | `float4` memory access     |
| PyTorch (cuDNN)  | reference      | Production baseline         |

> Numbers depend on GPU, row size, and batch size — run the same input shape across all kernels for a fair comparison.

## The kernels, in build order

**Stage 1 — Naive** (`Kernel/softmax_naive/`). One block per row, a single thread looping serially over the row. Deliberately slow; it exists to be obviously correct (the on-GPU correctness reference) and to give every later kernel something to beat.

**Stage 2 — Shared-memory reduction** (`Kernel/softmax_sharedmem/`). Each thread folds its strided slice of the row into a register, then the block finishes the max/sum with a classic shared-memory **tree reduction** (each step halves the active threads, turning an O(N) serial reduction into O(log N)). Note the `__syncthreads()` between every reduction step — forgetting it is the single most common bug in this whole project.

**Stage 3 — Warp-shuffle reduction** (`Kernel/softmax_warpshuffle/`). Replaces the shared-memory reduction with `__shfl_down_sync`, so the 32 threads of a warp exchange values directly through registers — no shared memory, no `__syncthreads()`. This is the reduction pattern PyTorch/cuDNN use internally, and the trickiest correctness stage.

**Stage 4 — Vectorized memory access** (`Kernel/softmax_vectorized/`). Softmax is memory-bound (little compute per byte), so the win here is bandwidth: each thread loads/stores a `float4` (16 bytes) per transaction instead of one float. Fast path needs `cols % 4 == 0`; a scalar tail handles any remainder.

**Stage 5 — Fused softmax + matmul** (`Kernel/softmax_fused/`, stretch goal). A mini scaled-dot-product attention head: `O = softmax(Q Kᵀ · scale) V`. One block owns one query row and keeps its scores in shared memory the whole time — the `Tq x Tk` score matrix is never written to global memory. This is the memory-traffic idea FlashAttention scales up; it's the direction the project points toward, not a full FlashAttention clone.

## Backward pass (training)

Forward softmax is inference only. To actually train through a softmax you need its gradient, so `Kernel/softmax_backward/` implements the backward pass. Given the forward output `y = softmax(x)` and the upstream gradient `dy = dL/dy`, the input gradient has a clean closed form:

```
dx_i = y_i * (dy_i - Σ_j y_j * dy_j)
```

Every row again reduces to a single reduction — the dot product `Σ_j y_j·dy_j` — followed by an elementwise combine, so the backward pass mirrors the forward optimization story exactly: a **naive** serial version, a **warp-shuffle** register reduction, and a **vectorized** `float4` version. Unlike the forward pass it uses no `exp()` and no max-subtraction (it consumes the already-normalized `y`), so it is numerically tame; correctness is validated both against a CPU reference and, independently, against a finite-difference gradient check. Run it with `make run-bwd`.

## Build & run

Requires the CUDA toolkit (`nvcc`). Pick your GPU architecture:

```bash
make ARCH=sm_75    # Turing (GTX 16xx, RTX 20xx, Colab/Kaggle T4)
make ARCH=sm_80    # A100
make ARCH=sm_86    # RTX 30-series (default)
make ARCH=sm_89    # RTX 40-series

./softmax_bench                 # forward:  default 4096 x 4096, 100 iters
./softmax_bench 8192 1024 200   # rows cols iters
./softmax_backward_bench        # backward (gradient) pass, same table format
```

`make` builds both benchmarks; `make run` runs the forward one and `make run-bwd` the backward one. Each driver checks every kernel against a CPU reference (`max abs error < 1e-3` → PASS) before timing it. The forward driver also runs the fused-attention correctness demo.

Add the production baseline row:

```bash
pip install torch
python bench/benchmark_pytorch.py --rows 4096 --cols 4096 --iters 100
```

### No GPU? Two options

- **Google Colab (recommended)** — free T4 GPU, full toolkit + PyTorch. See [`run_on_colab.ipynb`](run_on_colab.ipynb): it clones this repo, builds with `ARCH=sm_75`, and runs both benchmarks. Enable it via *Runtime → Change runtime type → T4 GPU*.
- **LeetGPU / in-browser runners** — paste [`standalone/softmax_all.cu`](standalone/softmax_all.cu) (the whole project in one file) for quick experimentation. No PyTorch baseline there.

## Correctness

Every kernel's output is compared element-wise against a numerically-stable CPU softmax (`utils/cpu_reference.cpp`); the fused kernel is compared against a CPU attention reference. The threshold is `1e-3` absolute error, which comfortably accommodates the fast-math `__expf` intrinsic and float32 accumulation order.

## Project layout

```
CUDA-Softmax/
├── Kernel/
│   ├── softmax_naive/          # one thread per row, serial reduction
│   ├── softmax_sharedmem/      # shared-memory tree reduction
│   ├── softmax_warpshuffle/    # __shfl_down_sync warp reduction
│   ├── softmax_vectorized/     # float4 loads/stores
│   ├── softmax_fused/          # fused softmax + matmul (mini attention)
│   └── softmax_backward/       # gradient pass: naive / warp / vectorized
├── Header/
│   └── softmax_kernels.cuh     # launcher decls + reduction helpers
├── utils/
│   ├── main.cu                 # forward correctness + benchmark driver
│   ├── backward_main.cu        # backward correctness + benchmark driver
│   ├── cpu_reference.cpp/.h    # ground-truth CPU softmax + attention + backward
├── bench/
│   └── benchmark_pytorch.py    # cuDNN production baseline
├── standalone/
│   └── softmax_all.cu          # single-file mirror (for LeetGPU etc.)
├── Benchmarks/                 # your results (git-ignored)
├── run_on_colab.ipynb          # build + run with a free GPU
└── Makefile
```

## Lessons learned

*(Fill this in as you go — reviewers love it.)* The warp-shuffle stage is usually the hardest: the block-level reduction has to combine per-warp partials through a second shuffle, and the `__syncthreads()` placement around the shared scratch buffer is easy to get subtly wrong.

## Reference reading

- NVIDIA CUDA C++ Programming Guide — memory hierarchy and warp primitives.
- Mark Harris, *Optimizing Parallel Reduction in CUDA* — the canonical reference for this exact problem.
- PyTorch/cuDNN softmax source — inspiration for the production approach.
- Dao et al., *FlashAttention* — only if attempting the Stage 5 fused stretch goal.

## License

MIT — see [LICENSE](LICENSE).
