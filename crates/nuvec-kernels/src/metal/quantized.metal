// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Affine quantization kernels (see quantized.h) for float16, bfloat16 and float32, group
// sizes 32, 64 and 128, and 1–6 or 8 bits, with the batched, partial-row and aligned
// variants MLX's dispatch selects, and the transposed gather (mixture-of-experts) kernels.

#include "steel_gemm.h"
#include "quantized.h"

#define NUVEC_Q_NAME(kind, iname, gs, b) "nuvec_affine_" #kind "_" #iname "_gs" #gs "_b" #b

#define NUVEC_Q(kind, iname, T, gs, b, suffix, ...)                                       \
    template [[host_name(NUVEC_Q_NAME(kind, iname, gs, b) suffix)]] [[kernel]]            \
    decltype(mlx::quant::affine_##kind<T, gs, b, ##__VA_ARGS__>)                           \
        mlx::quant::affine_##kind<T, gs, b, ##__VA_ARGS__>;

#define NUVEC_Q_ALL(iname, T, gs, b)                                                      \
    NUVEC_Q(quantize, iname, T, gs, b, "")                                                \
    NUVEC_Q(dequantize, iname, T, gs, b, "")                                              \
    NUVEC_Q(qmv_fast, iname, T, gs, b, "_batch0_pr0", false, false)                       \
    NUVEC_Q(qmv_fast, iname, T, gs, b, "_batch0_pr1", false, true)                        \
    NUVEC_Q(qmv_fast, iname, T, gs, b, "_batch1_pr0", true, false)                        \
    NUVEC_Q(qmv_fast, iname, T, gs, b, "_batch1_pr1", true, true)                         \
    NUVEC_Q(qmv, iname, T, gs, b, "_batch0", false)                                       \
    NUVEC_Q(qmv, iname, T, gs, b, "_batch1", true)                                        \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv1_batch0", 1, 8, false)                         \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv1_batch1", 1, 8, true)                          \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv2_batch0", 2, 8, false)                         \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv2_batch1", 2, 8, true)                          \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv3_batch0", 3, 8, false)                         \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv3_batch1", 3, 8, true)                          \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv4_batch0", 4, 8, false)                         \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv4_batch1", 4, 8, true)                          \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv5_batch0", 5, 8, false)                         \
    NUVEC_Q(qmv_wide, iname, T, gs, b, "_nv5_batch1", 5, 8, true)                          \
    NUVEC_Q(qvm, iname, T, gs, b, "_batch0", false)                                       \
    NUVEC_Q(qvm, iname, T, gs, b, "_batch1", true)                                        \
    NUVEC_Q(qvm_split_k, iname, T, gs, b, "_spk8", 8)                                     \
    NUVEC_Q(qvm_split_k, iname, T, gs, b, "_spk32", 32)                                   \
    NUVEC_Q(qmm_t, iname, T, gs, b, "_al0_batch0", false, false)                          \
    NUVEC_Q(qmm_t, iname, T, gs, b, "_al0_batch1", false, true)                           \
    NUVEC_Q(qmm_t, iname, T, gs, b, "_al1_batch0", true, false)                           \
    NUVEC_Q(qmm_t, iname, T, gs, b, "_al1_batch1", true, true)                            \
    NUVEC_Q(qmm_t_splitk, iname, T, gs, b, "_al0", false)                                 \
    NUVEC_Q(qmm_t_splitk, iname, T, gs, b, "_al1", true)                                  \
    NUVEC_Q(qmm_n, iname, T, gs, b, "_batch0", false)                                     \
    NUVEC_Q(qmm_n, iname, T, gs, b, "_batch1", true)                                      \
    NUVEC_Q(gather_qmv_fast, iname, T, gs, b, "")                                         \
    NUVEC_Q(gather_qmv, iname, T, gs, b, "")                                              \
    NUVEC_Q(gather_qmm_t, iname, T, gs, b, "_al0", false)                                 \
    NUVEC_Q(gather_qmm_t, iname, T, gs, b, "_al1", true)                                  \
    NUVEC_Q(gather_qmm_rhs, iname, T, gs, b, "_nt", 16, 32, 32, 1, 2, true)

#define NUVEC_Q_BITS(iname, T, gs)                                                        \
    NUVEC_Q_ALL(iname, T, gs, 1)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 2)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 3)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 4)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 5)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 6)                                                          \
    NUVEC_Q_ALL(iname, T, gs, 8)

#define NUVEC_Q_TYPE(iname, T)                                                            \
    NUVEC_Q_BITS(iname, T, 32)                                                            \
    NUVEC_Q_BITS(iname, T, 64)                                                            \
    NUVEC_Q_BITS(iname, T, 128)

NUVEC_Q_TYPE(float16, half)
NUVEC_Q_TYPE(bfloat16, bfloat16_t)
NUVEC_Q_TYPE(float32, float)

// MLX's `qmv_quad` (K of 64 or 128): the bit widths MLX routes to it (powers of two; 1 bit
// only at K = 128).
#define NUVEC_QMV_QUAD_D(iname, T, gs, b, d)                                              \
    NUVEC_Q(qmv_quad, iname, T, gs, b, "_d" #d "_batch0", d, false)                       \
    NUVEC_Q(qmv_quad, iname, T, gs, b, "_d" #d "_batch1", d, true)
#define NUVEC_QMV_QUAD(iname, T, gs)                                                      \
    NUVEC_QMV_QUAD_D(iname, T, gs, 1, 128)                                                \
    NUVEC_QMV_QUAD_D(iname, T, gs, 2, 64) NUVEC_QMV_QUAD_D(iname, T, gs, 2, 128)          \
    NUVEC_QMV_QUAD_D(iname, T, gs, 4, 64) NUVEC_QMV_QUAD_D(iname, T, gs, 4, 128)          \
    NUVEC_QMV_QUAD_D(iname, T, gs, 8, 64) NUVEC_QMV_QUAD_D(iname, T, gs, 8, 128)
#define NUVEC_QMV_QUAD_GS(iname, T)                                                       \
    NUVEC_QMV_QUAD(iname, T, 32) NUVEC_QMV_QUAD(iname, T, 64) NUVEC_QMV_QUAD(iname, T, 128)
NUVEC_QMV_QUAD_GS(float32, float)
NUVEC_QMV_QUAD_GS(float16, half)
NUVEC_QMV_QUAD_GS(bfloat16, bfloat16_t)

// nuvec's few-row products (quantized.h, `affine_qmv_rows`), every bit width, two or
// three rows per threadgroup (three for odd counts; four are slower).
#define NUVEC_QMV_ROWS_B(iname, T, gs, b)                                                \
    NUVEC_Q(qmv_rows, iname, T, gs, b, "_pr0_r2", false, 2)                                \
    NUVEC_Q(qmv_rows, iname, T, gs, b, "_pr1_r2", true, 2)                                 \
    NUVEC_Q(qmv_rows, iname, T, gs, b, "_pr0_r3", false, 3)                                \
    NUVEC_Q(qmv_rows, iname, T, gs, b, "_pr1_r3", true, 3)
#define NUVEC_QMV_ROWS(iname, T, gs)                                                     \
    NUVEC_QMV_ROWS_B(iname, T, gs, 1) NUVEC_QMV_ROWS_B(iname, T, gs, 2)                   \
    NUVEC_QMV_ROWS_B(iname, T, gs, 3) NUVEC_QMV_ROWS_B(iname, T, gs, 4)                   \
    NUVEC_QMV_ROWS_B(iname, T, gs, 5) NUVEC_QMV_ROWS_B(iname, T, gs, 6)                   \
    NUVEC_QMV_ROWS_B(iname, T, gs, 8)
#define NUVEC_QMV_ROWS_GS(iname, T) \
    NUVEC_QMV_ROWS(iname, T, 32) NUVEC_QMV_ROWS(iname, T, 64) NUVEC_QMV_ROWS(iname, T, 128)
NUVEC_QMV_ROWS_GS(float32, float)
NUVEC_QMV_ROWS_GS(float16, half)
NUVEC_QMV_ROWS_GS(bfloat16, bfloat16_t)
