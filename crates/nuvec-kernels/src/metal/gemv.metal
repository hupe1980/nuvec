// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Matrix-vector kernels: MLX's gemv and gemv_t (see gemv.h) for float16, bfloat16 and
// float32, with MLX's block shapes and the batched (nc) and axpby variants.

#include "steel_gemm.h"
#include "gemv.h"

#define NUVEC_GEMV(kind, iname, T, bm, bn, sm, sn, tm, tn, nc, axpby)                              \
    template [[host_name("nuvec_" #kind "_" #iname "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
                         "_tn" #tn "_nc" #nc "_axpby" #axpby)]] [[kernel]]                         \
    decltype(mlx::matvec::kind<T, bm, bn, sm, sn, tm, tn, bool(nc), bool(axpby)>)                    \
        mlx::matvec::kind<T, bm, bn, sm, sn, tm, tn, bool(nc), bool(axpby)>;

#define NUVEC_GEMV_VARIANTS(kind, iname, T, bm, bn, sm, sn, tm, tn)  \
    NUVEC_GEMV(kind, iname, T, bm, bn, sm, sn, tm, tn, 0, 0)          \
    NUVEC_GEMV(kind, iname, T, bm, bn, sm, sn, tm, tn, 0, 1)          \
    NUVEC_GEMV(kind, iname, T, bm, bn, sm, sn, tm, tn, 1, 0)          \
    NUVEC_GEMV(kind, iname, T, bm, bn, sm, sn, tm, tn, 1, 1)

#define NUVEC_GEMV_BLOCKS(iname, T)                                 \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 1, 8, 1, 32, 4, 4)          \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 1, 8, 1, 32, 1, 4)          \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 1, 1, 8, 4, 4, 4)           \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 1, 1, 8, 4, 1, 4)           \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 4, 1, 1, 32, 1, 4)          \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 4, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_VARIANTS(gemv, iname, T, 8, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_VARIANTS(gemv_t, iname, T, 1, 2, 8, 4, 4, 1)         \
    NUVEC_GEMV_VARIANTS(gemv_t, iname, T, 1, 2, 8, 4, 4, 4)         \
    NUVEC_GEMV_VARIANTS(gemv_t, iname, T, 1, 4, 8, 4, 4, 4)         \
    NUVEC_GEMV_VARIANTS(gemv_t, iname, T, 1, 16, 8, 4, 4, 4)        \
    NUVEC_GEMV_VARIANTS(gemv_t, iname, T, 1, 16, 4, 8, 4, 4)

NUVEC_GEMV_BLOCKS(float16, half)
NUVEC_GEMV_BLOCKS(bfloat16, bfloat16_t)
NUVEC_GEMV_BLOCKS(float32, float)
