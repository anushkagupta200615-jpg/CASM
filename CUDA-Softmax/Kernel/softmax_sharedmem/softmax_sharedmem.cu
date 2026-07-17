// ===========================================================================
//  Stage 2 -- SHARED-MEMORY tree reduction
//  One block per row, SOFTMAX_BLOCK threads cooperating. Each thread first
//  folds its strided slice of the row into a single register, then the block
//  finishes the max / sum with a classic shared-memory tree reduction
//  (each step halves the number of active threads). This is THE core idea of
//  the whole project: turning an O(N) serial reduction into O(log N) parallel.
//
//  Note: this stage deliberately uses NO warp shuffles -- that is stage 3.
//  Keeping them separate makes the speed-up from each technique measurable.
// ===========================================================================
#include "../../Header/softmax_kernels.cuh"

// Tree reduction over sdata[0..blockDim.x). Requires blockDim.x a power of two.
// Every step: the lower half absorbs the upper half. Result ends in sdata[0]
// and is returned to all threads.
__device__ __forceinline__ float treeReduceMax(float val, float* sdata) {
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();                      // <-- the classic "forgot this" bug
    }
    float r = sdata[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float treeReduceSum(float val, float* sdata) {
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float r = sdata[0];
    __syncthreads();
    return r;
}

__global__ void softmax_sharedmem_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int rows, int cols) {
    __shared__ float sdata[SOFTMAX_BLOCK];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    const float* x = in  + (size_t)row * cols;
    float*       y = out + (size_t)row * cols;

    // 1) each thread's partial max over a grid-stride slice of the row
    float local = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) local = fmaxf(local, x[j]);
    const float m = treeReduceMax(local, sdata);

    // 2) each thread's partial sum of exp(x - max)
    float lsum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) lsum += __expf(x[j] - m);
    const float sum = treeReduceSum(lsum, sdata);

    // 3) normalize
    const float inv = 1.0f / sum;
    for (int j = tid; j < cols; j += blockDim.x) y[j] = __expf(x[j] - m) * inv;
}

void launch_softmax_sharedmem(const float* d_in, float* d_out, int rows, int cols) {
    softmax_sharedmem_kernel<<<rows, SOFTMAX_BLOCK>>>(d_in, d_out, rows, cols);
}
