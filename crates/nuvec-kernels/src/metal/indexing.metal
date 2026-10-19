// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Indexing kernels: gather and scatter along an axis (`take_along_axis`, `put_along_axis`,
// scatter-add), and a strided copy into a region of the output (`concatenate`).
//
// Indices are any integer type. Negative indices count from the end of the axis;
// out-of-range indices are clamped (the CPU backend's rule). Elements are moved by size,
// except for scatter-add, which adds atomically (float16 and bfloat16 through a
// compare-and-swap loop on the containing 32-bit word, as MLX).

#include "common.h"

using namespace metal;
using namespace nuvec;

// The position an index selects on an axis of length `n`.
template <typename I> inline long axis_index(I raw, long n) {
    long v;
    if (I(-1) < I(0)) {
        v = long(raw);
        v = v < 0 ? v + n : v;
    } else {
        v = ulong(raw) >= ulong(n) ? n - 1 : long(raw);
    }
    return clamp(v, 0L, max(n - 1, 0L));
}

// Offsets of element `i` (row-major over `shape`) in two layouts; `a` skips `axis`.
inline void offsets(
    uint i,
    constant int* shape,
    constant long* a_strides,
    constant long* b_strides,
    int ndim,
    int axis,
    thread long& a,
    thread long& b) {
    a = 0;
    b = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        uint n = uint(shape[d]);
        long c = long(i % n);
        i /= n;
        b += c * b_strides[d];
        if (d != axis) {
            a += c * a_strides[d];
        }
    }
}

// out[i] = src[... index ...]: the output has the indices' shape.
template <typename T, typename I>
[[kernel]] void gather_axis(
    device const T* src [[buffer(0)]],
    device const I* ind [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant long& src_off [[buffer(3)]],
    constant long& ind_off [[buffer(4)]],
    constant int* shape [[buffer(5)]],
    constant long* src_strides [[buffer(6)]],
    constant long* ind_strides [[buffer(7)]],
    constant int& ndim [[buffer(8)]],
    constant int& axis [[buffer(9)]],
    constant long& axis_len [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {
    long s, k;
    offsets(i, shape, src_strides, ind_strides, ndim, axis, s, k);
    long j = axis_index(ind[ind_off + k], axis_len);
    out[i] = src[src_off + s + j * src_strides[axis]];
}

// Whole rows: `out[j, r] = src[ind[j], r]` for contiguous source rows of `row` elements
// (`take` along a leading axis), one thread per element on a 2-D grid, no index arithmetic.
template <typename T, typename I>
[[kernel]] void gather_rows(
    device const T* src [[buffer(0)]],
    device const I* ind [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant long& src_off [[buffer(3)]],
    constant long& ind_off [[buffer(4)]],
    constant long& ind_stride [[buffer(5)]],
    constant long& src_stride [[buffer(6)]],
    constant long& axis_len [[buffer(7)]],
    constant long& row [[buffer(8)]],
    uint2 p [[thread_position_in_grid]]) {
    long j = axis_index(ind[ind_off + long(p.y) * ind_stride], axis_len);
    out[long(p.y) * row + p.x] = src[src_off + j * src_stride + p.x];
}

// Atomic `*p += v`: native for 32-bit types; 16-bit floats through a compare-and-swap on
// the containing 32-bit word (adding in float and rounding once, as `op_add`).
inline void atomic_add(device float* p, float v) {
    atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed);
}
inline void atomic_add(device int* p, int v) {
    atomic_fetch_add_explicit((device atomic_int*)p, v, memory_order_relaxed);
}
inline void atomic_add(device uint* p, uint v) {
    atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
template <typename T> inline void atomic_add_16(device T* p, T v) {
    ulong address = ulong(p);
    uint shift = (address & 2) ? 16 : 0;
    device atomic_uint* word = (device atomic_uint*)(address & ~ulong(3));
    uint old = atomic_load_explicit(word, memory_order_relaxed);
    while (true) {
        T current = as_type<T>(ushort(old >> shift));
        T sum = op_add(current, v);
        uint next = (old & ~(0xFFFFu << shift)) | (uint(as_type<ushort>(sum)) << shift);
        if (atomic_compare_exchange_weak_explicit(
                word, &old, next, memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}
inline void atomic_add(device half* p, half v) { atomic_add_16(p, v); }
inline void atomic_add(device bfloat* p, bfloat v) { atomic_add_16(p, v); }

// Where update `i` lands in `out` (row-contiguous strides `out_strides`), and where it and
// its index are read.
inline void scatter_offsets(
    uint i,
    constant int* shape,
    constant long* upd_strides,
    constant long* ind_strides,
    constant long* out_strides,
    int ndim,
    int axis,
    thread long& u,
    thread long& k,
    thread long& o) {
    u = 0;
    k = 0;
    o = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        uint n = uint(shape[d]);
        long c = long(i % n);
        i /= n;
        u += c * upd_strides[d];
        k += c * ind_strides[d];
        if (d != axis) {
            o += c * out_strides[d];
        }
    }
}

// Writes each update into `out` (row-contiguous, already holding the destination) at its
// index.
template <typename T, typename I>
[[kernel]] void scatter_axis(
    device const T* upd [[buffer(0)]],
    device const I* ind [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant long& upd_off [[buffer(3)]],
    constant long& ind_off [[buffer(4)]],
    constant int* shape [[buffer(5)]],
    constant long* upd_strides [[buffer(6)]],
    constant long* ind_strides [[buffer(7)]],
    constant long* out_strides [[buffer(8)]],
    constant int& ndim [[buffer(9)]],
    constant int& axis [[buffer(10)]],
    constant long& axis_len [[buffer(11)]],
    uint i [[thread_position_in_grid]]) {
    long u, k, o;
    scatter_offsets(i, shape, upd_strides, ind_strides, out_strides, ndim, axis, u, k, o);
    long j = axis_index(ind[ind_off + k], axis_len);
    out[o + j * out_strides[axis]] = upd[upd_off + u];
}

// Adds each update to `out` at its index, atomically.
template <typename T, typename I>
[[kernel]] void scatter_add_axis(
    device const T* upd [[buffer(0)]],
    device const I* ind [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant long& upd_off [[buffer(3)]],
    constant long& ind_off [[buffer(4)]],
    constant int* shape [[buffer(5)]],
    constant long* upd_strides [[buffer(6)]],
    constant long* ind_strides [[buffer(7)]],
    constant long* out_strides [[buffer(8)]],
    constant int& ndim [[buffer(9)]],
    constant int& axis [[buffer(10)]],
    constant long& axis_len [[buffer(11)]],
    uint i [[thread_position_in_grid]]) {
    long u, k, o;
    scatter_offsets(i, shape, upd_strides, ind_strides, out_strides, ndim, axis, u, k, o);
    long j = axis_index(ind[ind_off + k], axis_len);
    atomic_add(out + o + j * out_strides[axis], upd[upd_off + u]);
}

// Copies an operand into a region of the output: `out[out_off + loc(i, out_strides)]`.
template <typename T>
[[kernel]] void copy_into(
    device const T* a [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant long& out_off [[buffer(3)]],
    constant int* shape [[buffer(4)]],
    constant long* a_strides [[buffer(5)]],
    constant long* out_strides [[buffer(6)]],
    constant int& ndim [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    out[out_off + loc(i, shape, out_strides, ndim)] = a[a_off + loc(i, shape, a_strides, ndim)];
}

// `copy_into` for at most three dimensions (collapsed for both layouts): a 3-D grid whose
// offsets are multiply-adds (MLX's `copy_gg_nd3`).
template <typename T>
[[kernel]] void copy_into_g3(
    device const T* a [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant long& a_off [[buffer(2)]],
    constant long& out_off [[buffer(3)]],
    constant int* shape [[buffer(4)]],
    constant long* a_strides [[buffer(5)]],
    constant long* out_strides [[buffer(6)]],
    constant int& ndim [[buffer(7)]],
    uint3 p [[thread_position_in_grid]]) {
    out[out_off + loc3(p, out_strides)] = a[a_off + loc3(p, a_strides)];
}

#define NUVEC_GATHER(bits, T, iname, I)                                                          \
    template [[host_name("nuvec_gather_" #bits "_" #iname)]] [[kernel]] void gather_axis<T, I>( \
        device const T*, device const I*, device T*, constant long&, constant long&,            \
        constant int*, constant long*, constant long*, constant int&, constant int&,            \
        constant long&, uint);                                                                  \
    template [[host_name("nuvec_gather_rows_" #bits "_" #iname)]] [[kernel]] void              \
    gather_rows<T, I>(device const T*, device const I*, device T*, constant long&,              \
        constant long&, constant long&, constant long&, constant long&, constant long&, uint2); \
    template [[host_name("nuvec_scatter_" #bits "_" #iname)]] [[kernel]] void                  \
    scatter_axis<T, I>(                                                                         \
        device const T*, device const I*, device T*, constant long&, constant long&,            \
        constant int*, constant long*, constant long*, constant long*, constant int&,           \
        constant int&, constant long&, uint);

#define NUVEC_SCATTER_ADD(dname, T, iname, I)                                                  \
    template [[host_name("nuvec_scatter_add_" #dname "_" #iname)]] [[kernel]] void            \
    scatter_add_axis<T, I>(                                                                     \
        device const T*, device const I*, device T*, constant long&, constant long&,            \
        constant int*, constant long*, constant long*, constant long*, constant int&,           \
        constant int&, constant long&, uint);

#define NUVEC_INDEXING(iname, I)                       \
    NUVEC_GATHER(8, uchar, iname, I)                   \
    NUVEC_GATHER(16, ushort, iname, I)                 \
    NUVEC_GATHER(32, uint, iname, I)                   \
    NUVEC_GATHER(64, ulong, iname, I)                  \
    NUVEC_SCATTER_ADD(float32, float, iname, I)        \
    NUVEC_SCATTER_ADD(float16, half, iname, I)         \
    NUVEC_SCATTER_ADD(bfloat16, bfloat, iname, I)      \
    NUVEC_SCATTER_ADD(int32, int, iname, I)            \
    NUVEC_SCATTER_ADD(uint32, uint, iname, I)

NUVEC_INDEXING(uint8, uchar)
NUVEC_INDEXING(uint16, ushort)
NUVEC_INDEXING(uint32, uint)
NUVEC_INDEXING(uint64, ulong)
NUVEC_INDEXING(int8, char)
NUVEC_INDEXING(int16, short)
NUVEC_INDEXING(int32, int)
NUVEC_INDEXING(int64, long)

#define NUVEC_COPY_INTO(bits, T)                                                               \
    template [[host_name("nuvec_copy_into_" #bits)]] [[kernel]] void copy_into<T>(            \
        device const T*, device T*, constant long&, constant long&, constant int*,             \
        constant long*, constant long*, constant int&, uint);                                   \
    template [[host_name("nuvec_copy_into_" #bits "_g3")]] [[kernel]] void copy_into_g3<T>(   \
        device const T*, device T*, constant long&, constant long&, constant int*,             \
        constant long*, constant long*, constant int&, uint3);

NUVEC_COPY_INTO(8, uchar)
NUVEC_COPY_INTO(16, ushort)
NUVEC_COPY_INTO(32, uint)
NUVEC_COPY_INTO(64, ulong)

// 16-byte units, for layouts whose innermost dimension is contiguous and aligned.
template [[host_name("nuvec_copy_into_128_g3")]] [[kernel]] void copy_into_g3<uint4>(
    device const uint4*, device uint4*, constant long&, constant long&, constant int*,
    constant long*, constant long*, constant int&, uint3);
