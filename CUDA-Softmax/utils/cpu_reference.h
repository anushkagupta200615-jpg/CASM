// ===========================================================================
//  cpu_reference.h
//  Ground-truth CPU implementations used to validate every GPU kernel.
//  If a kernel disagrees with these beyond a small float tolerance, it's wrong
//  -- optimize nothing until the naive kernel matches this exactly.
// ===========================================================================
#pragma once

// Numerically-stable row-wise softmax on the CPU. [rows x cols], row-major.
void softmax_cpu(const float* in, float* out, int rows, int cols);

// Reference scaled-dot-product attention: O = softmax(Q Kᵀ * scale) V.
//   Q [Tq x d], K [Tk x d], V [Tk x d], O [Tq x d].
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int Tq, int Tk, int d, float scale);

// Largest absolute element-wise difference between two arrays of length n.
float max_abs_error(const float* a, const float* b, int n);
