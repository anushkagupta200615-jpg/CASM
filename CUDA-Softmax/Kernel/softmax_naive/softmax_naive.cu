// ===========================================================================
//  Stage 1 -- NAIVE softmax
//  One thread block per row, ONE thread doing all the work serially.
//  Deliberately slow. Its only jobs: (a) be obviously correct, so it can serve
//  as the on-GPU correctness reference, and (b) give every later kernel a
//  baseline to beat.
// ===========================================================================
#include "../../Header/softmax_kernels.cuh"
#include <cstdio>

__global__ void softmax_naive_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int rows, int cols) {
    const int row = blockIdx.x;               // one block == one row
    if (row >= rows) return;
    if (threadIdx.x != 0) return;             // single worker thread

    const float* x = in  + (size_t)row * cols;
    float*       y = out + (size_t)row * cols;

    // pass 1: serial max
    float m = -INFINITY;
    for (int j = 0; j < cols; ++j) m = fmaxf(m, x[j]);

    // pass 2: serial sum of exp(x - max)
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) { y[j] = __expf(x[j] - m); sum += y[j]; }

    // pass 3: normalize
    const float inv = 1.0f / sum;
    for (int j = 0; j < cols; ++j) y[j] *= inv;
}

void launch_softmax_naive(const float* d_in, float* d_out, int rows, int cols) {
    // block of 1 thread; the kernel ignores threadIdx.x != 0 anyway.
    softmax_naive_kernel<<<rows, 1>>>(d_in, d_out, rows, cols);
}
