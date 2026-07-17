// ===========================================================================
//  cpu_reference.cpp   -- see cpu_reference.h
// ===========================================================================
#include "cpu_reference.h"
#include <cmath>
#include <vector>

void softmax_cpu(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* x = in  + (size_t)r * cols;
        float*       y = out + (size_t)r * cols;

        // 1) row max (the numerical-stability shift)
        float m = -INFINITY;
        for (int j = 0; j < cols; ++j) m = std::fmax(m, x[j]);

        // 2) exponentiate the shifted values and accumulate the denominator
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            y[j] = std::exp(x[j] - m);
            sum += y[j];
        }

        // 3) normalize
        const float inv = 1.0f / sum;
        for (int j = 0; j < cols; ++j) y[j] *= inv;
    }
}

void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int Tq, int Tk, int d, float scale) {
    std::vector<float> scores(Tk);
    for (int q = 0; q < Tq; ++q) {
        // scores = Q[q] . K[k]  * scale
        float m = -INFINITY;
        for (int k = 0; k < Tk; ++k) {
            float dot = 0.0f;
            for (int i = 0; i < d; ++i) dot += Q[(size_t)q * d + i] * K[(size_t)k * d + i];
            scores[k] = dot * scale;
            m = std::fmax(m, scores[k]);
        }
        // softmax over the Tk scores
        float sum = 0.0f;
        for (int k = 0; k < Tk; ++k) { scores[k] = std::exp(scores[k] - m); sum += scores[k]; }
        const float inv = 1.0f / sum;
        // O[q] = sum_k softmax_k * V[k]
        for (int i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < Tk; ++k) acc += scores[k] * V[(size_t)k * d + i];
            O[(size_t)q * d + i] = acc * inv;
        }
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float e = 0.0f;
    for (int i = 0; i < n; ++i) e = std::fmax(e, std::fabs(a[i] - b[i]));
    return e;
}
