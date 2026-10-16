// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Batched matrix multiplication: a 16x16 output tile per threadgroup, K stepped through
// threadgroup memory, float32 accumulation. Operands are fully strided (row and column
// strides, plus batch strides), so transposed and broadcast operands need no copies.
// A simple, correct baseline; the tuned GEMM is a later feature.

#include "common.h"

using namespace metal;
using namespace nuvec;

constant constexpr uint TILE = 16;

template <typename T>
[[kernel]] void matmul(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int* mnk [[buffer(3)]],              // M, N, K
    constant long* a_mat [[buffer(4)]],           // offset, row stride, column stride
    constant long* b_mat [[buffer(5)]],
    constant int* batch_shape [[buffer(6)]],
    constant long* a_batch [[buffer(7)]],
    constant long* b_batch [[buffer(8)]],
    constant int& batch_ndim [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[TILE][TILE];
    threadgroup float b_tile[TILE][TILE];
    const int m = mnk[0], n = mnk[1], k = mnk[2];
    const int row = int(group.y * TILE + lid.y);
    const int col = int(group.x * TILE + lid.x);
    const long a_base = a_mat[0] + loc(group.z, batch_shape, a_batch, batch_ndim);
    const long b_base = b_mat[0] + loc(group.z, batch_shape, b_batch, batch_ndim);
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += int(TILE)) {
        int ak = k0 + int(lid.x);
        int bk = k0 + int(lid.y);
        a_tile[lid.y][lid.x] = (row < m && ak < k) ? float(a[a_base + row * a_mat[1] + ak * a_mat[2]]) : 0.0f;
        b_tile[lid.y][lid.x] = (bk < k && col < n) ? float(b[b_base + bk * b_mat[1] + col * b_mat[2]]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE; ++kk) {
            acc = fma(a_tile[lid.y][kk], b_tile[kk][lid.x], acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) {
        out[long(group.z) * m * n + long(row) * n + col] = T(acc);
    }
}

#define NUVEC_MATMUL(name, T)                                                                    \
    template [[host_name("nuvec_matmul_" #name)]] [[kernel]] void matmul<T>(                     \
        device const T*, device const T*, device T*, constant int*, constant long*, constant long*, \
        constant int*, constant long*, constant long*, constant int&, uint3, uint3);

NUVEC_MATMUL(float16, half)
NUVEC_MATMUL(bfloat16, bfloat)
NUVEC_MATMUL(float32, float)
