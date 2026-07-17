// ===========================================================================
//  main.cu  --  correctness + benchmark driver for all kernels
//
//  For a fixed input shape it:
//    1. builds a random [rows x cols] matrix,
//    2. runs each softmax kernel, checking max-abs-error vs the CPU reference,
//    3. times each kernel with CUDA events (warmup + N iterations),
//    4. prints a relative-speed table (naive = 1.0x baseline),
//    5. runs the fused mini-attention demo and checks it vs the CPU reference.
//
//  Build with the Makefile (links every Kernel/*.cu + cpu_reference.cpp).
//  Usage:  ./softmax_bench [rows] [cols] [iters]
// ===========================================================================
#include "../Header/softmax_kernels.cuh"
#include "cpu_reference.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// ---- tiny deterministic RNG so results are reproducible run to run ----------
static float frand(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    // map to roughly [-4, 4): a range wide enough that a missing max-subtraction
    // would overflow expf(), so the stability trick is actually exercised.
    return ((state >> 8) / (float)(1u << 24)) * 8.0f - 4.0f;
}

struct KernelEntry {
    const char* name;
    void (*launch)(const float*, float*, int, int);
    const char* technique;
};

static float time_kernel(void (*launch)(const float*, float*, int, int),
                         const float* d_in, float* d_out,
                         int rows, int cols, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup (also surfaces launch errors before timing)
    for (int i = 0; i < 5; ++i) launch(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;                        // average ms per call
}

// ---------------------------------------------------------------------------
static void run_fused_demo() {
    // small single-head attention: Tq queries, Tk keys, head dim d
    const int Tq = 128, Tk = 256, d = 64;
    const float scale = 1.0f / std::sqrt((float)d);

    std::vector<float> hQ((size_t)Tq * d), hK((size_t)Tk * d), hV((size_t)Tk * d);
    std::vector<float> hO((size_t)Tq * d), hRef((size_t)Tq * d);
    unsigned s = 987654321u;
    for (auto& v : hQ) v = frand(s);
    for (auto& v : hK) v = frand(s);
    for (auto& v : hV) v = frand(s);

    attention_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), Tq, Tk, d, scale);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, hO.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_fused_attention(dQ, dK, dV, dO, Tq, Tk, d, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_abs_error(hO.data(), hRef.data(), (int)hO.size());
    printf("\nStage 5 -- fused mini-attention (Tq=%d, Tk=%d, d=%d)\n", Tq, Tk, d);
    printf("  max abs error vs CPU: %.3e  ->  %s\n", err, err < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int rows  = (argc > 1) ? atoi(argv[1]) : 4096;
    int cols  = (argc > 2) ? atoi(argv[2]) : 4096;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Shape : %d rows x %d cols   iters=%d\n\n", rows, cols, iters);

    const size_t N = (size_t)rows * cols;
    std::vector<float> h_in(N), h_out(N), h_ref(N);
    unsigned s = 123456789u;
    for (size_t i = 0; i < N; ++i) h_in[i] = frand(s);

    // CPU ground truth
    softmax_cpu(h_in.data(), h_ref.data(), rows, cols);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    KernelEntry kernels[] = {
        {"Naive",         launch_softmax_naive,      "Serial reduction"},
        {"Shared Memory", launch_softmax_sharedmem,  "Tree reduction"},
        {"Warp Shuffle",  launch_softmax_warpshuffle,"Register-level reduction"},
        {"Vectorized",    launch_softmax_vectorized, "float4 memory access"},
    };
    const int K = sizeof(kernels) / sizeof(kernels[0]);

    float times[8]; bool ok[8];
    for (int k = 0; k < K; ++k) {
        // correctness
        CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
        kernels[k].launch(d_in, d_out, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_abs_error(h_out.data(), h_ref.data(), (int)N);
        ok[k] = err < 1e-3f;

        // timing
        times[k] = time_kernel(kernels[k].launch, d_in, d_out, rows, cols, iters);
        printf("  %-14s  max_err=%.3e  %s\n", kernels[k].name, err, ok[k] ? "PASS" : "FAIL");
    }

    // ---- results table -----------------------------------------------------
    printf("\n| %-14s | %-12s | %-11s | %-24s |\n", "Kernel", "ms/call", "Speedup", "Key Technique");
    printf("|%s|%s|%s|%s|\n",
           "----------------", "--------------", "-------------", "--------------------------");
    const float base = times[0];
    for (int k = 0; k < K; ++k) {
        printf("| %-14s | %10.4f   | %8.2fx   | %-24s |\n",
               kernels[k].name, times[k], base / times[k], kernels[k].technique);
    }
    printf("\n(Run bench/benchmark_pytorch.py to add the PyTorch/cuDNN reference row.)\n");

    // ---- fused stretch demo ------------------------------------------------
    run_fused_demo();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
