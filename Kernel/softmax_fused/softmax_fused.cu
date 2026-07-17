// ===========================================================================
//  Stage 5 (stretch) -- FUSED softmax + matmul  ==  mini attention
//  Computes one attention head:  O = softmax( (Q Kᵀ) * scale ) V
//
//  The point is FUSION: a naive implementation would (1) matmul Q Kᵀ into a
//  [Tq x Tk] score matrix in global memory, (2) read it back to softmax it,
//  (3) read it again to multiply by V. Here one block owns one query row and
//  keeps its Tk scores in shared memory the whole time -- the score matrix is
//  never written to global memory. That is exactly the memory-traffic idea
//  FlashAttention scales up. This is "the direction the project points
//  toward", not a full FlashAttention clone.
//
//  Kept intentionally small/fixed: Tk must fit in shared memory.
// ===========================================================================
#include "../../Header/softmax_kernels.cuh"

__global__ void fused_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int Tq, int Tk, int d, float scale) {
    extern __shared__ float scores[];         // dynamic: Tk floats
    __shared__ float red[32];                 // scratch for block reduction
    const int q   = blockIdx.x;               // one block == one query row
    if (q >= Tq) return;
    const int tid = threadIdx.x;

    const float* Qrow = Q + (size_t)q * d;

    // 1) scores[k] = (Q[q] . K[k]) * scale     -- the Q Kᵀ matmul, fused in
    for (int k = tid; k < Tk; k += blockDim.x) {
        const float* Krow = K + (size_t)k * d;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) dot += Qrow[i] * Krow[i];
        scores[k] = dot * scale;
    }
    __syncthreads();

    // 2) softmax over the Tk scores, in shared memory (numerically stable)
    float local = -INFINITY;
    for (int k = tid; k < Tk; k += blockDim.x) local = fmaxf(local, scores[k]);
    const float m = blockReduceMax(local, red);

    float lsum = 0.0f;
    for (int k = tid; k < Tk; k += blockDim.x) {
        float e = __expf(scores[k] - m);
        scores[k] = e;                        // overwrite score with its exp
        lsum += e;
    }
    __syncthreads();                          // all exp writes visible
    const float sum = blockReduceSum(lsum, red);
    const float inv = 1.0f / sum;

    // 3) O[q, :] = sum_k softmax_k * V[k, :]   -- the second matmul, fused in
    for (int i = tid; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < Tk; ++k) acc += scores[k] * V[(size_t)k * d + i];
        O[(size_t)q * d + i] = acc * inv;
    }
}

void launch_fused_attention(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int Tq, int Tk, int d, float scale) {
    const int block  = SOFTMAX_BLOCK;
    const size_t smem = (size_t)Tk * sizeof(float);   // scores live here
    fused_attention_kernel<<<Tq, block, smem>>>(d_Q, d_K, d_V, d_O, Tq, Tk, d, scale);
}
