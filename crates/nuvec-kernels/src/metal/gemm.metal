// SPDX-License-Identifier: MIT OR Apache-2.0
//
// GEMM kernels: MLX's steel GEMM (see steel_gemm.h) for float16, bfloat16 and float32, with
// MLX's six tile shapes and the four transpose variants. Alignment, batching and the AddMM
// epilogue are function constants (10, 100, 110, 200-202), as in MLX. Split-K kernels
// (partial products in float32, then an accumulation kernel) cover few output tiles over a
// long K.

#include "steel_gemm.h"

#define NUVEC_GEMM(tname, ta, tb, iname, T, bm, bn, bk, wm, wn)                                     \
    template [[host_name("nuvec_gemm_" #tname "_" #iname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm \
                         "_wn" #wn)]] [[kernel]] decltype(mlx::steel::gemm<T, bm, bn, bk, wm, wn,  \
                                                                         ta, tb, float>)          \
        mlx::steel::gemm<T, bm, bn, bk, wm, wn, ta, tb, float>;

#define NUVEC_GEMM_TRANSPOSES(iname, T, bm, bn, bk, wm, wn)          \
    NUVEC_GEMM(nn, false, false, iname, T, bm, bn, bk, wm, wn)       \
    NUVEC_GEMM(nt, false, true, iname, T, bm, bn, bk, wm, wn)        \
    NUVEC_GEMM(tn, true, false, iname, T, bm, bn, bk, wm, wn)        \
    NUVEC_GEMM(tt, true, true, iname, T, bm, bn, bk, wm, wn)

#define NUVEC_GEMM_SHAPES(iname, T)                      \
    NUVEC_GEMM_TRANSPOSES(iname, T, 64, 64, 16, 2, 2)    \
    NUVEC_GEMM_TRANSPOSES(iname, T, 64, 64, 16, 1, 2)    \
    NUVEC_GEMM_TRANSPOSES(iname, T, 64, 32, 32, 2, 2)    \
    NUVEC_GEMM_TRANSPOSES(iname, T, 32, 64, 16, 1, 2)    \
    NUVEC_GEMM_TRANSPOSES(iname, T, 32, 32, 16, 2, 2)    \
    NUVEC_GEMM_TRANSPOSES(iname, T, 64, 32, 8, 4, 1)

NUVEC_GEMM_SHAPES(float16, half)
NUVEC_GEMM_SHAPES(bfloat16, bfloat16_t)
NUVEC_GEMM_SHAPES(float32, float)

#define NUVEC_SPLITK(tname, ta, tb, iname, T, bm, bn, mn, mnb, kn, kb)                              \
    template [[host_name("nuvec_gemm_splitk_" #tname "_" #iname "_bm" #bm "_bn" #bn "_MN_" #mn     \
                         "_K_" #kn)]] [[kernel]] decltype(mlx::steel::gemm_splitk<T, float, bm, bn, \
                                                                                16, 2, 2, ta, tb,  \
                                                                                mnb, kb>)          \
        mlx::steel::gemm_splitk<T, float, bm, bn, 16, 2, 2, ta, tb, mnb, kb>;

#define NUVEC_SPLITK_ALIGNED(tname, ta, tb, iname, T, bm, bn)              \
    NUVEC_SPLITK(tname, ta, tb, iname, T, bm, bn, t, true, t, true)          \
    NUVEC_SPLITK(tname, ta, tb, iname, T, bm, bn, t, true, n, false)         \
    NUVEC_SPLITK(tname, ta, tb, iname, T, bm, bn, n, false, t, true)         \
    NUVEC_SPLITK(tname, ta, tb, iname, T, bm, bn, n, false, n, false)

#define NUVEC_SPLITK_TRANSPOSES(iname, T, bm, bn)                    \
    NUVEC_SPLITK_ALIGNED(nn, false, false, iname, T, bm, bn)         \
    NUVEC_SPLITK_ALIGNED(nt, false, true, iname, T, bm, bn)          \
    NUVEC_SPLITK_ALIGNED(tn, true, false, iname, T, bm, bn)          \
    NUVEC_SPLITK_ALIGNED(tt, true, true, iname, T, bm, bn)

#define NUVEC_SPLITK_SHAPES(iname, T)                \
    NUVEC_SPLITK_TRANSPOSES(iname, T, 16, 16)        \
    NUVEC_SPLITK_TRANSPOSES(iname, T, 16, 32)        \
    NUVEC_SPLITK_TRANSPOSES(iname, T, 32, 16)        \
    NUVEC_SPLITK_TRANSPOSES(iname, T, 32, 32)

NUVEC_SPLITK_SHAPES(float16, half)
NUVEC_SPLITK_SHAPES(bfloat16, bfloat16_t)
NUVEC_SPLITK_SHAPES(float32, float)

#define NUVEC_SPLITK_ACCUM(oname, T)                                                                 \
    template [[host_name("nuvec_gemm_splitk_accum_" #oname)]] [[kernel]] decltype(                 \
        mlx::steel::gemm_splitk_accum<float, T>) mlx::steel::gemm_splitk_accum<float, T>;          \
    template [[host_name("nuvec_gemm_splitk_accum_axpby_" #oname)]] [[kernel]] decltype(           \
        mlx::steel::gemm_splitk_accum_axpby<float, T>) mlx::steel::gemm_splitk_accum_axpby<float, T>;

NUVEC_SPLITK_ACCUM(float16, half)
NUVEC_SPLITK_ACCUM(bfloat16, bfloat16_t)
NUVEC_SPLITK_ACCUM(float32, float)
