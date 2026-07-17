// ===========================================================================
//  Stage 4 -- VECTORIZED memory access (float4)
//  Softmax is memory-bound: very little arithmetic per byte read. So the win
//  here is bandwidth, not math. Each thread loads/stores a float4 (16 bytes)
//  per transaction instead of a single float, cutting the number of memory
//  instructions ~4x. Reduction is the same warp-shuffle block reduce as
//  stage 3.
//
//  Fast path needs cols % 4 == 0 and 16-byte-aligned rows (guaranteed when
//  cols % 4 == 0 and the buffer comes from cudaMalloc). A scalar tail handles
//  any leftover 1-3 columns so the kernel stays correct for arbitrary cols.
// ===========================================================================
#include "../../Header/softmax_kernels.cuh"

__global__ void softmax_vectorized_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int rows, int cols) {
    __shared__ float shared[32];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    const float* xrow = in  + (size_t)row * cols;
    float*       yrow = out + (size_t)row * cols;

    const int nvec   = cols >> 2;             // number of full float4 groups
    const int tail0  = nvec << 2;             // first scalar (remainder) index
    const float4* xv = reinterpret_cast<const float4*>(xrow);
    float4*       yv = reinterpret_cast<float4*>(yrow);

    // ----- 1) row max -----
    float local = -INFINITY;
    for (int i = tid; i < nvec; i += blockDim.x) {
        float4 v = xv[i];
        local = fmaxf(local, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    for (int j = tail0 + tid; j < cols; j += blockDim.x) local = fmaxf(local, xrow[j]);
    const float m = blockReduceMax(local, shared);

    // ----- 2) denominator -----
    float lsum = 0.0f;
    for (int i = tid; i < nvec; i += blockDim.x) {
        float4 v = xv[i];
        lsum += __expf(v.x - m) + __expf(v.y - m) + __expf(v.z - m) + __expf(v.w - m);
    }
    for (int j = tail0 + tid; j < cols; j += blockDim.x) lsum += __expf(xrow[j] - m);
    const float sum = blockReduceSum(lsum, shared);

    // ----- 3) normalize (vectorized store) -----
    const float inv = 1.0f / sum;
    for (int i = tid; i < nvec; i += blockDim.x) {
        float4 v = xv[i], o;
        o.x = __expf(v.x - m) * inv;
        o.y = __expf(v.y - m) * inv;
        o.z = __expf(v.z - m) * inv;
        o.w = __expf(v.w - m) * inv;
        yv[i] = o;
    }
    for (int j = tail0 + tid; j < cols; j += blockDim.x) yrow[j] = __expf(xrow[j] - m) * inv;
}

void launch_softmax_vectorized(const float* d_in, float* d_out, int rows, int cols) {
    softmax_vectorized_kernel<<<rows, SOFTMAX_BLOCK>>>(d_in, d_out, rows, cols);
}
