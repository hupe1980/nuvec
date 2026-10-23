// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Matrix-vector kernels: MLX's gemv and gemv_t (see gemv.h) for float16, bfloat16 and
// float32, with MLX's block shapes and the batched (nc) and axpby variants; and nuvec's
// multi-row variants (`gemv_rows`, `gemv_t_rows`: up to R vectors per call, the matrix
// read once per group of up to R = 2, 4 or 8 rows, each row's arithmetic the one-row
// kernel's; groups of one tile run side by side).

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

#define NUVEC_GEMV_ROWS(kind, iname, T, bm, bn, sm, sn, tm, tn, r, nc, axpby)                    \
    template [[host_name("nuvec_" #kind "_rows" #r "_" #iname "_bm" #bm "_bn" #bn "_sm" #sm "_sn" \
                         #sn "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby)]] [[kernel]]         \
    decltype(mlx::matvec::kind##_rows<T, bm, bn, sm, sn, tm, tn, r, bool(nc), bool(axpby)>)      \
        mlx::matvec::kind##_rows<T, bm, bn, sm, sn, tm, tn, r, bool(nc), bool(axpby)>;

#define NUVEC_GEMV_ROWS_R(kind, iname, T, bm, bn, sm, sn, tm, tn, r)    \
    NUVEC_GEMV_ROWS(kind, iname, T, bm, bn, sm, sn, tm, tn, r, 0, 0)      \
    NUVEC_GEMV_ROWS(kind, iname, T, bm, bn, sm, sn, tm, tn, r, 0, 1)      \
    NUVEC_GEMV_ROWS(kind, iname, T, bm, bn, sm, sn, tm, tn, r, 1, 0)      \
    NUVEC_GEMV_ROWS(kind, iname, T, bm, bn, sm, sn, tm, tn, r, 1, 1)

#define NUVEC_GEMV_ROWS_VARIANTS(kind, iname, T, bm, bn, sm, sn, tm, tn) \
    NUVEC_GEMV_ROWS_R(kind, iname, T, bm, bn, sm, sn, tm, tn, 2)          \
    NUVEC_GEMV_ROWS_R(kind, iname, T, bm, bn, sm, sn, tm, tn, 4)          \
    NUVEC_GEMV_ROWS_R(kind, iname, T, bm, bn, sm, sn, tm, tn, 8)

#define NUVEC_GEMV_ROWS_BLOCKS(iname, T)                                 \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 1, 8, 1, 32, 4, 4)          \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 1, 8, 1, 32, 1, 4)          \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 1, 1, 8, 4, 4, 4)           \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 1, 1, 8, 4, 1, 4)           \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 4, 1, 1, 32, 1, 4)          \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 4, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_ROWS_VARIANTS(gemv, iname, T, 8, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_ROWS_VARIANTS(gemv_t, iname, T, 1, 2, 8, 4, 4, 1)         \
    NUVEC_GEMV_ROWS_VARIANTS(gemv_t, iname, T, 1, 2, 8, 4, 4, 4)         \
    NUVEC_GEMV_ROWS_VARIANTS(gemv_t, iname, T, 1, 4, 8, 4, 4, 4)         \
    NUVEC_GEMV_ROWS_VARIANTS(gemv_t, iname, T, 1, 16, 8, 4, 4, 4)        \
    NUVEC_GEMV_ROWS_VARIANTS(gemv_t, iname, T, 1, 16, 4, 8, 4, 4)

NUVEC_GEMV_ROWS_BLOCKS(float16, half)
NUVEC_GEMV_ROWS_BLOCKS(bfloat16, bfloat16_t)
NUVEC_GEMV_ROWS_BLOCKS(float32, float)

// MLX's gathered matrix-vector products (`gemv_gather`, gemv.h) for `gather_mm`.
#define NUVEC_GEMV_GATHER(kind, t, iname, T, bm, bn, sm, sn, tm, tn)                          \
    template [[host_name("nuvec_" #kind "_gather_" #iname "_bm" #bm "_bn" #bn "_sm" #sm "_sn" \
                         #sn "_tm" #tm "_tn" #tn)]] [[kernel]]                               \
    decltype(mlx::matvec::gemv_gather<T, bm, bn, sm, sn, tm, tn, t>)                          \
        mlx::matvec::gemv_gather<T, bm, bn, sm, sn, tm, tn, t>;

#define NUVEC_GEMV_GATHER_BLOCKS(iname, T)                               \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 1, 8, 1, 32, 4, 4)          \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 1, 8, 1, 32, 1, 4)          \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 1, 1, 8, 4, 4, 4)           \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 1, 1, 8, 4, 1, 4)           \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 4, 1, 1, 32, 1, 4)          \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 4, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_GATHER(gemv, false, iname, T, 8, 1, 1, 32, 4, 4)          \
    NUVEC_GEMV_GATHER(gemv_t, true, iname, T, 1, 2, 8, 4, 4, 1)          \
    NUVEC_GEMV_GATHER(gemv_t, true, iname, T, 1, 2, 8, 4, 4, 4)          \
    NUVEC_GEMV_GATHER(gemv_t, true, iname, T, 1, 4, 8, 4, 4, 4)          \
    NUVEC_GEMV_GATHER(gemv_t, true, iname, T, 1, 16, 8, 4, 4, 4)         \
    NUVEC_GEMV_GATHER(gemv_t, true, iname, T, 1, 16, 4, 8, 4, 4)

NUVEC_GEMV_GATHER_BLOCKS(float16, half)
NUVEC_GEMV_GATHER_BLOCKS(bfloat16, bfloat16_t)
NUVEC_GEMV_GATHER_BLOCKS(float32, float)
