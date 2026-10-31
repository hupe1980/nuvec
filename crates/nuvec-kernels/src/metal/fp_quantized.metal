// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Floating-point quantization kernels (see fp_quantized.h) for the mxfp4 (groups of 32,
// 4 bits), mxfp8 (32, 8 bits) and nvfp4 (16, 4 bits) modes and float16, bfloat16 and
// float32: the variants MLX's dispatch selects, with global-scale (`_hgs`) gathers for nvfp4.

#include "steel_gemm.h"
#include "fp_quantized.h"

#define NUVEC_FP_NAME(mode, kind, iname) "nuvec_" #mode "_" #kind "_" #iname

#define NUVEC_FP(mode, kind, iname, T, gs, b, suffix, ...)                                \
    template [[host_name(NUVEC_FP_NAME(mode, kind, iname) suffix)]] [[kernel]]            \
    decltype(mlx::fpq::fp_##kind<T, gs, b, ##__VA_ARGS__>)                                 \
        mlx::fpq::fp_##kind<T, gs, b, ##__VA_ARGS__>;

// Kernels of every mode (no global scale).
#define NUVEC_FP_ALL(mode, iname, T, gs, b)                                               \
    NUVEC_FP(mode, quantize, iname, T, gs, b, "")                                         \
    NUVEC_FP(mode, dequantize, iname, T, gs, b, "")                                       \
    NUVEC_FP(mode, qmv_quad, iname, T, gs, b, "_d64_batch0", 64, false)                   \
    NUVEC_FP(mode, qmv_quad, iname, T, gs, b, "_d64_batch1", 64, true)                    \
    NUVEC_FP(mode, qmv_quad, iname, T, gs, b, "_d128_batch0", 128, false)                 \
    NUVEC_FP(mode, qmv_quad, iname, T, gs, b, "_d128_batch1", 128, true)                  \
    NUVEC_FP(mode, qmv_fast, iname, T, gs, b, "_batch0_pr0_r4", false, false, false, 4)   \
    NUVEC_FP(mode, qmv_fast, iname, T, gs, b, "_batch0_pr1_r4", false, true, false, 4)    \
    NUVEC_FP(mode, qmv_fast, iname, T, gs, b, "_batch1_pr0_r4", true, false, false, 4)    \
    NUVEC_FP(mode, qmv_fast, iname, T, gs, b, "_batch1_pr1_r4", true, true, false, 4)     \
    NUVEC_FP(mode, qmv, iname, T, gs, b, "_batch0", false)                                \
    NUVEC_FP(mode, qmv, iname, T, gs, b, "_batch1", true)                                 \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv2_batch0", 2, 16, false)                \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv2_batch1", 2, 16, true)                 \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv3_batch0", 3, 16, false)                \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv3_batch1", 3, 16, true)                 \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv4_batch0", 4, 16, false)                \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv4_batch1", 4, 16, true)                 \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv5_batch0", 5, 16, false)                \
    NUVEC_FP(mode, qmv_wide, iname, T, gs, b, "_nv5_batch1", 5, 16, true)                 \
    NUVEC_FP(mode, qvm, iname, T, gs, b, "_batch0", false)                                \
    NUVEC_FP(mode, qvm, iname, T, gs, b, "_batch1", true)                                 \
    NUVEC_FP(mode, qvm_split_k, iname, T, gs, b, "_spk8", 8)                              \
    NUVEC_FP(mode, qvm_split_k, iname, T, gs, b, "_spk32", 32)                            \
    NUVEC_FP(mode, qmm_t, iname, T, gs, b, "_al0_batch0", false, false)                   \
    NUVEC_FP(mode, qmm_t, iname, T, gs, b, "_al0_batch1", false, true)                    \
    NUVEC_FP(mode, qmm_t, iname, T, gs, b, "_al1_batch0", true, false)                    \
    NUVEC_FP(mode, qmm_t, iname, T, gs, b, "_al1_batch1", true, true)                     \
    NUVEC_FP(mode, qmm_t_splitk, iname, T, gs, b, "_al0", false)                          \
    NUVEC_FP(mode, qmm_t_splitk, iname, T, gs, b, "_al1", true)                           \
    NUVEC_FP(mode, qmm_n, iname, T, gs, b, "_batch0", false)                              \
    NUVEC_FP(mode, qmm_n, iname, T, gs, b, "_batch1", true)                               \
    NUVEC_FP(mode, gather_qmv_fast, iname, T, gs, b, "", false)                           \
    NUVEC_FP(mode, gather_qmv, iname, T, gs, b, "", false)                                \
    NUVEC_FP(mode, gather_qvm, iname, T, gs, b, "", false)                                \
    NUVEC_FP(mode, gather_qmm_t, iname, T, gs, b, "_al0", false, false)                   \
    NUVEC_FP(mode, gather_qmm_t, iname, T, gs, b, "_al1", true, false)                    \
    NUVEC_FP(mode, gather_qmm_n, iname, T, gs, b, "", false)                              \
    NUVEC_FP(mode, gather_qmm_rhs, iname, T, gs, b, "_nt", 16, 32, 32, 1, 2, true, false) \
    NUVEC_FP(mode, gather_qmm_rhs, iname, T, gs, b, "_nn", 16, 32, 32, 1, 2, false, false) \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast1_pr0_r2", true, false, 2)            \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast1_pr1_r2", true, true, 2)             \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast1_pr0_r4", true, false, 4)            \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast1_pr1_r4", true, true, 4)             \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast0_pr0_r2", false, false, 2)           \
    NUVEC_FP(mode, qmv_rows, iname, T, gs, b, "_fast0_pr0_r4", false, false, 4)

// nvfp4 only: the global-scale gathers and the narrow `qmv_fast` (M5, MLX's `r_2`).
#define NUVEC_FP_NV(iname, T)                                                             \
    NUVEC_FP(nvfp4, gather_qmv_fast, iname, T, 16, 4, "_hgs", true)                       \
    NUVEC_FP(nvfp4, gather_qmv, iname, T, 16, 4, "_hgs", true)                            \
    NUVEC_FP(nvfp4, gather_qvm, iname, T, 16, 4, "_hgs", true)                            \
    NUVEC_FP(nvfp4, gather_qmm_t, iname, T, 16, 4, "_al0_hgs", false, true)               \
    NUVEC_FP(nvfp4, gather_qmm_t, iname, T, 16, 4, "_al1_hgs", true, true)                \
    NUVEC_FP(nvfp4, gather_qmm_n, iname, T, 16, 4, "_hgs", true)                          \
    NUVEC_FP(nvfp4, gather_qmm_rhs, iname, T, 16, 4, "_nt_hgs", 16, 32, 32, 1, 2, true, true) \
    NUVEC_FP(nvfp4, gather_qmm_rhs, iname, T, 16, 4, "_nn_hgs", 16, 32, 32, 1, 2, false, true) \
    NUVEC_FP(nvfp4, qmv_fast, iname, T, 16, 4, "_batch0_pr0_r2", false, false, false, 2)  \
    NUVEC_FP(nvfp4, qmv_fast, iname, T, 16, 4, "_batch0_pr1_r2", false, true, false, 2)   \
    NUVEC_FP(nvfp4, qmv_fast, iname, T, 16, 4, "_batch1_pr0_r2", true, false, false, 2)   \
    NUVEC_FP(nvfp4, qmv_fast, iname, T, 16, 4, "_batch1_pr1_r2", true, true, false, 2)

#define NUVEC_FP_TYPE(iname, T)                                                           \
    NUVEC_FP_ALL(mxfp4, iname, T, 32, 4)                                                  \
    NUVEC_FP_ALL(mxfp8, iname, T, 32, 8)                                                  \
    NUVEC_FP_ALL(nvfp4, iname, T, 16, 4)                                                  \
    NUVEC_FP_NV(iname, T)

NUVEC_FP_TYPE(float16, half)
NUVEC_FP_TYPE(bfloat16, bfloat16_t)
NUVEC_FP_TYPE(float32, float)
