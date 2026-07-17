// ===========================================================================
//  Stage 3 -- WARP-SHUFFLE reduction
//  Same one-block-per-row layout, but the reduction now happens through
//  registers using __shfl_down_sync (see blockReduce* in the header). Threads
//  inside a warp exchange values directly; only the handful of per-warp
//  partials ever touch shared memory. This is the reduction pattern PyTorch /
//  cuDNN actually use internally.
//
//  Trickiest correctness stage -- diff its output against stage 2 carefully.
// ===========================================================================
#include "../../Header/softmax_kernels.cuh"

__global__ void softmax_warpshuffle_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int rows, int cols) {
    __shared__ float shared[32];              // one slot per warp (<=32 warps)
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    const float* x = in  + (size_t)row * cols;
    float*       y = out + (size_t)row * cols;

    // 1) partial max -> block max via warp shuffles
    float local = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) local = fmaxf(local, x[j]);
    const float m = blockReduceMax(local, shared);

    // 2) partial sum -> block sum
    float lsum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) lsum += __expf(x[j] - m);
    const float sum = blockReduceSum(lsum, shared);

    // 3) normalize
    const float inv = 1.0f / sum;
    for (int j = tid; j < cols; j += blockDim.x) y[j] = __expf(x[j] - m) * inv;
}

void launch_softmax_warpshuffle(const float* d_in, float* d_out, int rows, int cols) {
    softmax_warpshuffle_kernel<<<rows, SOFTMAX_BLOCK>>>(d_in, d_out, rows, cols);
}
