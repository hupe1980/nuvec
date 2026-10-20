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
    NUVEC_Q(qvm, iname, T, gs, b, "_batch0", false)                                       \
    NUVEC_Q(qvm, iname, T, gs, b, "_batch1", true)                                        \
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

// nuvec's few-row 4-bit products (quantized.h, `affine_qmv_rows`), two rows per
// threadgroup (more are slower).
#define NUVEC_QMV_ROWS(iname, T, gs)                                                     \
    NUVEC_Q(qmv_rows, iname, T, gs, 4, "_pr0_r2", false, 2)                                \
    NUVEC_Q(qmv_rows, iname, T, gs, 4, "_pr1_r2", true, 2)
#define NUVEC_QMV_ROWS_GS(iname, T) \
    NUVEC_QMV_ROWS(iname, T, 32) NUVEC_QMV_ROWS(iname, T, 64) NUVEC_QMV_ROWS(iname, T, 128)
NUVEC_QMV_ROWS_GS(float32, float)
NUVEC_QMV_ROWS_GS(float16, half)
NUVEC_QMV_ROWS_GS(bfloat16, bfloat16_t)
