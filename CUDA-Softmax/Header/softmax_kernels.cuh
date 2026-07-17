// ===========================================================================
//  softmax_kernels.cuh
//  Shared declarations + device-side reduction helpers for every kernel stage.
//
//  Design contract for the "plain softmax" kernels (stages 1-4):
//    * Input  : row-major matrix  d_in  [rows x cols]  (float)
//    * Output : row-major matrix  d_out [rows x cols]  (float)
//    * Launch : ONE thread block per row. Each block reduces its own row.
//    * Every kernel subtracts the row max before exp() -> numerically stable.
// ===========================================================================
#pragma once

#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------------------ config --
// Fixed block size for the parallel kernels. Must be a power of two because
// the shared-memory tree reduction (stage 2) halves the active thread count
// each step. 256 is a good default: enough parallelism, low occupancy cost.
#ifndef SOFTMAX_BLOCK
#define SOFTMAX_BLOCK 256
#endif

#define FULL_MASK 0xffffffffu   // all 32 lanes participate in the shuffle

// ------------------------------------------------------- error-check macro --
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ===========================================================================
//  Warp-level primitives  (stage 3+)
//  __shfl_down_sync lets the 32 threads of a warp exchange registers directly
//  -- no shared memory, no __syncthreads(). After the loop, LANE 0 holds the
//  reduced value for the warp.
// ===========================================================================
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
//  Block reduction built on warp shuffles (stage 3 / 4).
//  Every thread contributes `val`; every thread receives the block-wide result.
//  `shared` must point to at least (blockDim.x / 32) floats (<= 32).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float blockReduceMax(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int numWarps = (blockDim.x + 31) >> 5;

    val = warpReduceMax(val);                 // reduce within each warp
    if (lane == 0) shared[wid] = val;         // lane 0 publishes its warp result
    __syncthreads();

    // Let warp 0 pull the per-warp partials and finish the reduction.
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -INFINITY;
    __syncthreads();                          // all reads done before overwrite
    if (wid == 0) {
        val = warpReduceMax(val);
        if (lane == 0) shared[0] = val;       // broadcast slot
    }
    __syncthreads();
    float result = shared[0];
    __syncthreads();                          // safe to reuse `shared` after
    return result;
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int numWarps = (blockDim.x + 31) >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    __syncthreads();
    if (wid == 0) {
        val = warpReduceSum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}

// ===========================================================================
//  Host-side launcher declarations. Each is implemented in its own .cu file
//  under Kernel/, so the pieces compile independently and link together.
// ===========================================================================
void launch_softmax_naive     (const float* d_in, float* d_out, int rows, int cols);
void launch_softmax_sharedmem (const float* d_in, float* d_out, int rows, int cols);
void launch_softmax_warpshuffle(const float* d_in, float* d_out, int rows, int cols);
void launch_softmax_vectorized(const float* d_in, float* d_out, int rows, int cols);

// Stage 5 (stretch): fused scaled-dot-product attention.
//   Q,K,V : [Tq x d], [Tk x d], [Tk x d]   ->  O : [Tq x d]
//   Computes  O = softmax( (Q Kᵀ) * scale ) V   without ever writing the
//   [Tq x Tk] score matrix to global memory.
void launch_fused_attention(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int Tq, int Tk, int d, float scale);
