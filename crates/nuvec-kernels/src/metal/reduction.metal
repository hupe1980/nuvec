// SPDX-License-Identifier: MIT
//
// MLX's reductions, ported from MLX v0.32.4 (mlx/backend/metal/kernels/reduction/*.h, the
// operators of ops.h without their unused atomic updates, and LoopedElemToLoc from utils.h)
// into `mlx::red`: one-pass and two-pass all-reductions, row reductions (small, simple,
// looped) and column reductions (small, long-column, looped, two-pass). Instantiated with
// 32-bit indexing for the types nuvec reduces on the GPU (sums and products of widened
// integers and floats, extremes of every integer and float type, and/or of bools). Sums and
// products of float16 and bfloat16 accumulate and write float32 (MLX accumulates in the
// input type); the caller rounds the result.
//
// Copyright © 2023-2024 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "fast.h"

static constant constexpr int REDUCE_N_READS = 4;
static constant constexpr int REDUCE_N_WRITES = 4;

namespace mlx {
namespace red {

template <typename U, typename T>
inline U cast_to(T val) {
  return static_cast<U>(val);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    IdxT elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) thread : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int* shape, const constant int64_t* strides) thread {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides)
      thread {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() thread {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) thread : dim(dim) {}

  void next(const constant int* shape, const constant int64_t* strides) thread {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides)
      thread {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() thread {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) thread {}

  void next(const constant int*, const constant int64_t* strides) thread {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int*, const constant int64_t* strides)
      thread {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() thread {
    return offset;
  }
};

// SIMD functions Metal lacks (from MLX's utils.h and bf16_math.h): 64-bit shuffles as two
// 32-bit lanes, and bfloat16 reductions computed in float.
inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bfloat16_t simd_shuffle_down(bfloat16_t data, uint16_t delta) {
  return as_type<bfloat16_t>(metal::simd_shuffle_down(as_type<ushort>(data), delta));
}

#define NUVEC_BF16_SIMD(name)                                       \
  inline bfloat16_t name(bfloat16_t data) {                         \
    return static_cast<bfloat16_t>(metal::name(static_cast<float>(data))); \
  }
NUVEC_BF16_SIMD(simd_sum)
NUVEC_BF16_SIMD(simd_product)
NUVEC_BF16_SIMD(simd_max)
NUVEC_BF16_SIMD(simd_min)

using metal::simd_max;
using metal::simd_min;
using metal::simd_product;
using metal::simd_shuffle_down;
using metal::simd_sum;

#define DEFINE_SIMD_REDUCE()                                             \
  template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>  \
  T simd_reduce(T val) thread {                                          \
    return simd_reduce_impl(val);                                        \
  }                                                                      \
                                                                         \
  template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true> \
  T simd_reduce(T val) thread {                                          \
    for (short i = simd_size / 2; i > 0; i /= 2) {                       \
      val = operator()(val, simd_shuffle_down(val, i));                  \
    }                                                                    \
    return val;                                                          \
  }

static constant constexpr const uint8_t simd_size = 32;

// MLX's reduction operators (kernels/reduction/ops.h) without the atomic updates, which
// the ported dispatch never uses.
template <typename U = bool>
struct And {
  DEFINE_SIMD_REDUCE()
  bool simd_reduce_impl(bool val) thread {
    return simd_all(val);
  }
  static constexpr constant bool init = true;
  bool operator()(bool a, bool b) thread {
    return a && b;
  }
};

template <typename U = bool>
struct Or {
  DEFINE_SIMD_REDUCE()
  bool simd_reduce_impl(bool val) thread {
    return simd_any(val);
  }
  static constexpr constant bool init = false;
  bool operator()(bool a, bool b) thread {
    return a || b;
  }
};

template <typename U>
struct Sum {
  DEFINE_SIMD_REDUCE()
  template <typename T>
  T simd_reduce_impl(T val) thread {
    return simd_sum(val);
  }
  static constexpr constant U init = U(0);
  U operator()(U a, U b) thread {
    return a + b;
  }
};

template <typename U>
struct Prod {
  DEFINE_SIMD_REDUCE()
  template <typename T>
  T simd_reduce_impl(T val) thread {
    return simd_product(val);
  }
  static constexpr constant U init = U(1);
  U operator()(U a, U b) thread {
    return a * b;
  }
};

template <typename U>
struct Min {
  DEFINE_SIMD_REDUCE()
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) thread {
    return simd_min(val);
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) thread {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_min(val);
  }
  static constexpr constant U init = Limits<U>::max;
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    return a < b ? a : b;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    }
    return a < b ? a : b;
  }
};

template <typename U>
struct Max {
  DEFINE_SIMD_REDUCE()
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val) thread {
    return simd_max(val);
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> simd_reduce_impl(T val) thread {
    if (simd_any(val != val)) {
      return static_cast<T>(NAN);
    }
    return simd_max(val);
  }
  static constexpr constant U init = Limits<U>::min;
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    return a > b ? a : b;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T a, T b) thread {
    if (metal::isnan(a) || metal::isnan(b)) {
      return static_cast<T>(NAN);
    }
    return a > b ? a : b;
  }
};

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& in_size [[buffer(2)]],
    const constant size_t& row_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];

  U total = Op::init;
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;

  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }

  for (IdxT b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) {
      total = op(cast_to<U>(in[i]), total);
    }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) {
      total = op(cast_to<U>(in[i]), total);
    }
  }

  // Reduction within simd group
  total = op.simd_reduce(total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_vals[simd_group_id] = total;
    }

    // Reduction within thread group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total = op.simd_reduce(total);
  }

  if (lid.x == 0) {
    out[gid.y] = total;
  }
}

// Row reduction utilities
// - `per_thread_row_reduce` collaborative partial reduction in the threadgroup
// - `threadgroup_reduce` collaborative reduction in the threadgroup such that
//   lid.x == 0 holds the reduced value
// - `thread_reduce` simple loop and reduce the row

/**
 * The thread group collaboratively reduces across the rows with bounds
 * checking. In the end each thread holds a part of the reduction.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* inputs[N_WRITES],
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  Op op;

  // Set up the accumulator registers
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  // Loop over the reduction size within thread group
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(cast_to<U>(inputs[j][i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  // Separate case for the last set as we close the reduction size
  int index = lid_x * N_READS;
  if (index + N_READS <= extra) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(cast_to<U>(inputs[j][i]), totals[j]);
      }
    }
  } else {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; index + i < extra; i++) {
        totals[j] = op(cast_to<U>(inputs[j][i]), totals[j]);
      }
    }
  }
}

/**
 * Consecutive rows in a contiguous array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const constant size_t& reduction_size,
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Consecutive rows in an arbitrarily ordered array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const int64_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant int64_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Reduce within the threadgroup.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  // Simdgroup first
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Across simdgroups
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }

    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC void
thread_reduce(thread U& total, const device T* row, int blocks, int extra) {
  Op op;
  for (int i = 0; i < blocks; i++) {
    U vals[N_READS];
    for (int j = 0; j < N_READS; j++) {
      vals[j] = row[j];
    }
    for (int j = 0; j < N_READS; j++) {
      total = op(vals[j], total);
    }
    row += N_READS;
  }
  for (int i = 0; i < extra; i++) {
    total = op(*row++, total);
  }
}

// Reduction kernels
// - `row_reduce_small` depending on the non-row reductions and row size it
//   either just loops over everything or a simd collaboratively reduces the
//   non_row reductions. In the first case one thread is responsible for one
//   output on the 2nd one simd is responsible for one output.
// - `row_reduce_simple` simple contiguous row reduction
// - `row_reduce_looped` simply loop and reduce each row for each non-row
//   reduction. One threadgroup is responsible for one output.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;

  U total_val = Op::init;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);

  // Precompute some row reduction numbers
  const device T* row;
  int blocks = IdxT(row_size) / N_READS;
  int extra = IdxT(row_size) % N_READS;

  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    // Simple loop over non_row_reductions and reduce the row in the thread.
    IdxT out_idx = tid.x + tsize.x * IdxT(tid.y);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }

    out[out_idx] = total_val;
  } else {
    // Collaboratively reduce over non_row_reductions in the simdgroup. Each
    // thread reduces every 32nd row and then a simple simd reduce.
    IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    loop.next(simd_lane_id, reduce_shape, reduce_strides);

    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    total_val = op.simd_reduce(total_val);

    if (simd_lane_id == 0) {
      out[out_idx] = total_val;
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];

  // Move to the row
  IdxT out_idx = N_WRITES * (gid.y + gsize.y * IdxT(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * IdxT(reduction_size);
  out += out_idx;

  // Each thread reduces across the row
  int blocks = IdxT(reduction_size) / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      out[i] = totals[i];
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);

  // lid.x * N_READS breaks the per_thread_row_reduce interface a bit. Maybe it
  // needs a small refactor.
  in +=
      elem_to_loc<IdxT>(out_idx, shape, strides, ndim) + IdxT(lid.x) * N_READS;

  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;
  int blocks = IdxT(row_size) / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);

  for (IdxT i = 0; i < non_row_reductions; i++) {
    row = in + loop.location();

    // Each thread reduces across the row
    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);

    // Aggregate across rows
    total = op(total, row_total);

    loop.next(reduce_shape, reduce_strides);
  }

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  constexpr int n_reads = 4;
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  U totals[n_reads];
  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  IdxT column = IdxT(gid.x) * lsize.x * n_reads + lid.x * n_reads;
  if (column >= reduction_stride) {
    return;
  }
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(lid.y, reduce_shape, reduce_strides);
  for (IdxT r = lid.y; r < total_rows; r += lsize.y) {
    row = in + loop.location();
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(cast_to<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? cast_to<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }
    loop.next(lsize.y, reduce_shape, reduce_strides);
  }

  if (lsize.y > 1) {
    // lsize.y should be <= 8
    threadgroup U shared_vals[32 * 8 * n_reads];
    for (int i = 0; i < n_reads; i++) {
      shared_vals[lid.y * lsize.x * n_reads + lid.x * n_reads + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = shared_vals[lid.x * n_reads + i];
      }
      for (uint j = 1; j < lsize.y; j++) {
        for (int i = 0; i < n_reads; i++) {
          totals[i] =
              op(shared_vals[j * lsize.x * n_reads + lid.x * n_reads + i],
                 totals[i]);
        }
      }
    }
  }

  if (lid.y == 0) {
    out += out_idx * IdxT(reduction_stride) + column;
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_longcolumn(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  IdxT out_idx = gid.x + gsize.x * IdxT(gid.y);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + IdxT(lid.x);

  U total = Op::init;
  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(gid.z * lsize.y + lid.y, reduce_shape, reduce_strides);
  for (IdxT r = gid.z * lsize.y + lid.y; r < total_rows;
       r += lsize.y * gsize.z) {
    row = in + loop.location();
    total = op(cast_to<U>(*row), total);
    loop.next(lsize.y * gsize.z, reduce_shape, reduce_strides);
  }

  threadgroup U shared_vals[32 * 32];
  shared_vals[lid.y * lsize.x + lid.x] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0) {
    for (uint i = 1; i < lsize.y; i++) {
      total = op(total, shared_vals[i * lsize.x + lid.x]);
    }
    out[gid.z * IdxT(out_size) + out_idx * IdxT(reduction_stride) + lid.x] =
        total;
  }
}

/**
 * Our approach is the following simple looped approach:
 *  1. Each thread keeps running totals for BN / n_simdgroups outputs.
 *  2. Load a tile BM, BN in registers and accumulate in the running totals
 *  3. Move ahead by BM steps until the column axis and the non column
 *     reductions are exhausted.
 *  6. If BM == 32 then transpose in SM and simd reduce the running totals.
 *     Otherwise write in shared memory and BN threads accumulate the running
 *     totals with a loop.
 *  7. Write them to the output
 */
template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant int64_t& column_offset [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = IdxT(BN) * gid.x + IdxT(column_offset) + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (IdxT r = offset.y; r < total; r += BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(cast_to<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? cast_to<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  if (BM == 32) {
    constexpr int n_outputs = BN / n_simdgroups;
    static_assert(
        BM != 32 || n_outputs == n_reads,
        "The tile should be selected such that n_outputs == n_reads");
    for (int i = 0; i < n_reads; i++) {
      shared_vals[offset.y * BN + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
    for (int i = 0; i < n_outputs; i++) {
      totals[i] =
          op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
    }

    // Write the output.
    if (simd_lane_id == 0) {
      IdxT out_column = IdxT(BN) * gid.x + IdxT(column_offset) + out_offset.x;
      out += out_idx * IdxT(reduction_stride) + out_column;
      if (out_column + n_outputs <= reduction_stride) {
        for (int i = 0; i < n_outputs; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; out_column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }

  // Each thread holds n_reads partial results. We write them all out to shared
  // memory and threads with offset.y == 0 aggregate the columns and write the
  // outputs.
  else {
    short x_block = offset.x / n_reads;
    for (int i = 0; i < n_reads; i++) {
      shared_vals[x_block * BM * n_reads + i * BM + offset.y] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (offset.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        for (int j = 1; j < BM; j++) {
          totals[i] =
              op(shared_vals[x_block * BM * n_reads + i * BM + j], totals[i]);
        }
      }
    }

    // Write the output.
    if (offset.y == 0) {
      out += out_idx * IdxT(reduction_stride) + column;
      if (safe) {
        for (int i = 0; i < n_reads; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_2pass(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant int64_t& column_offset [[buffer(11)]],
    const constant size_t& out_size [[buffer(12)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;
  constexpr int n_outputs = BN / n_simdgroups;
  constexpr short outer_blocks = 32;
  static_assert(BM == 32, "BM should be equal to 32");

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = IdxT(BN) * gid.x + IdxT(column_offset) + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT full_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT block_idx = full_idx / IdxT(out_size);
  IdxT out_idx = full_idx % IdxT(out_size);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y + block_idx * BM, reduce_shape, reduce_strides);
  for (IdxT r = offset.y + block_idx * BM; r < total; r += outer_blocks * BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(cast_to<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? cast_to<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(outer_blocks * BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  for (int i = 0; i < n_reads; i++) {
    shared_vals[offset.y * BN + offset.x + i] = totals[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
  for (int i = 0; i < n_outputs; i++) {
    totals[i] =
        op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
  }

  // Write the output.
  if (simd_lane_id == 0) {
    IdxT out_column = IdxT(BN) * gid.x + IdxT(column_offset) + out_offset.x;
    out += full_idx * IdxT(reduction_stride) + out_column;
    if (out_column + n_outputs <= reduction_stride) {
      for (int i = 0; i < n_outputs; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; out_column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}

} // namespace red
} // namespace mlx

template [[host_name("nuvec_all_reduce_and_bool")]] [[kernel]] decltype(mlx::red::all_reduce<bool, bool, mlx::red::And<bool>, int64_t>) mlx::red::all_reduce<bool, bool, mlx::red::And<bool>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bool, bool, mlx::red::And<bool>, size_t>) mlx::red::row_reduce_simple<bool, bool, mlx::red::And<bool>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 1>) mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 1>) mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 1>) mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 1>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 2>) mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 2>) mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 2>) mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 2>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 5>) mlx::red::row_reduce_small<bool, bool, mlx::red::And<bool>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_and_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 5>) mlx::red::row_reduce_looped<bool, bool, mlx::red::And<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 5>) mlx::red::col_reduce_small<bool, bool, mlx::red::And<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 5>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::And<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::And<bool>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_and_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::And<bool>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_or_bool")]] [[kernel]] decltype(mlx::red::all_reduce<bool, bool, mlx::red::Or<bool>, int64_t>) mlx::red::all_reduce<bool, bool, mlx::red::Or<bool>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bool, bool, mlx::red::Or<bool>, size_t>) mlx::red::row_reduce_simple<bool, bool, mlx::red::Or<bool>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 1>) mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 1>) mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 1>) mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 1>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 2>) mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 2>) mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 2>) mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 2>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 5>) mlx::red::row_reduce_small<bool, bool, mlx::red::Or<bool>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_or_bool")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 5>) mlx::red::row_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 5>) mlx::red::col_reduce_small<bool, bool, mlx::red::Or<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 5>) mlx::red::col_reduce_longcolumn<bool, bool, mlx::red::Or<bool>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bool, bool, mlx::red::Or<bool>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_or_bool")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bool, bool, mlx::red::Or<bool>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_int32")]] [[kernel]] decltype(mlx::red::all_reduce<int32_t, int32_t, mlx::red::Sum<int32_t>, int64_t>) mlx::red::all_reduce<int32_t, int32_t, mlx::red::Sum<int32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Sum<int32_t>, size_t>) mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Sum<int32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Sum<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_int64")]] [[kernel]] decltype(mlx::red::all_reduce<int64_t, int64_t, mlx::red::Sum<int64_t>, int64_t>) mlx::red::all_reduce<int64_t, int64_t, mlx::red::Sum<int64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Sum<int64_t>, size_t>) mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Sum<int64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Sum<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_uint32")]] [[kernel]] decltype(mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int64_t>) mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, size_t>) mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Sum<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_uint64")]] [[kernel]] decltype(mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int64_t>) mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, size_t>) mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Sum<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_float16")]] [[kernel]] decltype(mlx::red::all_reduce<half, float, mlx::red::Sum<float>, int64_t>) mlx::red::all_reduce<half, float, mlx::red::Sum<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<half, float, mlx::red::Sum<float>, size_t>) mlx::red::row_reduce_simple<half, float, mlx::red::Sum<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_small<half, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_looped<half, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_small<half, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_bfloat16")]] [[kernel]] decltype(mlx::red::all_reduce<bfloat16_t, float, mlx::red::Sum<float>, int64_t>) mlx::red::all_reduce<bfloat16_t, float, mlx::red::Sum<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bfloat16_t, float, mlx::red::Sum<float>, size_t>) mlx::red::row_reduce_simple<bfloat16_t, float, mlx::red::Sum<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_sum_float32")]] [[kernel]] decltype(mlx::red::all_reduce<float, float, mlx::red::Sum<float>, int64_t>) mlx::red::all_reduce<float, float, mlx::red::Sum<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<float, float, mlx::red::Sum<float>, size_t>) mlx::red::row_reduce_simple<float, float, mlx::red::Sum<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 1>) mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 1>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 2>) mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 2>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_small<float, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_sum_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 5>) mlx::red::row_reduce_looped<float, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_small<float, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 5>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Sum<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_sum_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Sum<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_int32")]] [[kernel]] decltype(mlx::red::all_reduce<int32_t, int32_t, mlx::red::Prod<int32_t>, int64_t>) mlx::red::all_reduce<int32_t, int32_t, mlx::red::Prod<int32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Prod<int32_t>, size_t>) mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Prod<int32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Prod<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_int64")]] [[kernel]] decltype(mlx::red::all_reduce<int64_t, int64_t, mlx::red::Prod<int64_t>, int64_t>) mlx::red::all_reduce<int64_t, int64_t, mlx::red::Prod<int64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Prod<int64_t>, size_t>) mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Prod<int64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Prod<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_uint32")]] [[kernel]] decltype(mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int64_t>) mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, size_t>) mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Prod<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_uint64")]] [[kernel]] decltype(mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int64_t>) mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, size_t>) mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Prod<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_float16")]] [[kernel]] decltype(mlx::red::all_reduce<half, float, mlx::red::Prod<float>, int64_t>) mlx::red::all_reduce<half, float, mlx::red::Prod<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<half, float, mlx::red::Prod<float>, size_t>) mlx::red::row_reduce_simple<half, float, mlx::red::Prod<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_small<half, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_looped<half, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_small<half, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_longcolumn<half, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<half, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<half, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_bfloat16")]] [[kernel]] decltype(mlx::red::all_reduce<bfloat16_t, float, mlx::red::Prod<float>, int64_t>) mlx::red::all_reduce<bfloat16_t, float, mlx::red::Prod<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bfloat16_t, float, mlx::red::Prod<float>, size_t>) mlx::red::row_reduce_simple<bfloat16_t, float, mlx::red::Prod<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_small<bfloat16_t, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_longcolumn<bfloat16_t, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_prod_float32")]] [[kernel]] decltype(mlx::red::all_reduce<float, float, mlx::red::Prod<float>, int64_t>) mlx::red::all_reduce<float, float, mlx::red::Prod<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<float, float, mlx::red::Prod<float>, size_t>) mlx::red::row_reduce_simple<float, float, mlx::red::Prod<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 1>) mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 1>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 2>) mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 2>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_small<float, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_prod_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 5>) mlx::red::row_reduce_looped<float, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_small<float, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 5>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Prod<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_prod_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Prod<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_int8")]] [[kernel]] decltype(mlx::red::all_reduce<int8_t, int8_t, mlx::red::Max<int8_t>, int64_t>) mlx::red::all_reduce<int8_t, int8_t, mlx::red::Max<int8_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int8_t, int8_t, mlx::red::Max<int8_t>, size_t>) mlx::red::row_reduce_simple<int8_t, int8_t, mlx::red::Max<int8_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Max<int8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_int16")]] [[kernel]] decltype(mlx::red::all_reduce<int16_t, int16_t, mlx::red::Max<int16_t>, int64_t>) mlx::red::all_reduce<int16_t, int16_t, mlx::red::Max<int16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int16_t, int16_t, mlx::red::Max<int16_t>, size_t>) mlx::red::row_reduce_simple<int16_t, int16_t, mlx::red::Max<int16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Max<int16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_int32")]] [[kernel]] decltype(mlx::red::all_reduce<int32_t, int32_t, mlx::red::Max<int32_t>, int64_t>) mlx::red::all_reduce<int32_t, int32_t, mlx::red::Max<int32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Max<int32_t>, size_t>) mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Max<int32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Max<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_int64")]] [[kernel]] decltype(mlx::red::all_reduce<int64_t, int64_t, mlx::red::Max<int64_t>, int64_t>) mlx::red::all_reduce<int64_t, int64_t, mlx::red::Max<int64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Max<int64_t>, size_t>) mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Max<int64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Max<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_uint8")]] [[kernel]] decltype(mlx::red::all_reduce<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int64_t>) mlx::red::all_reduce<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint8_t, uint8_t, mlx::red::Max<uint8_t>, size_t>) mlx::red::row_reduce_simple<uint8_t, uint8_t, mlx::red::Max<uint8_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Max<uint8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_uint16")]] [[kernel]] decltype(mlx::red::all_reduce<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int64_t>) mlx::red::all_reduce<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint16_t, uint16_t, mlx::red::Max<uint16_t>, size_t>) mlx::red::row_reduce_simple<uint16_t, uint16_t, mlx::red::Max<uint16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Max<uint16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_uint32")]] [[kernel]] decltype(mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int64_t>) mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Max<uint32_t>, size_t>) mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Max<uint32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Max<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_uint64")]] [[kernel]] decltype(mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int64_t>) mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Max<uint64_t>, size_t>) mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Max<uint64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Max<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_float16")]] [[kernel]] decltype(mlx::red::all_reduce<half, half, mlx::red::Max<half>, int64_t>) mlx::red::all_reduce<half, half, mlx::red::Max<half>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<half, half, mlx::red::Max<half>, size_t>) mlx::red::row_reduce_simple<half, half, mlx::red::Max<half>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 1>) mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 1>) mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 1>) mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 1>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 1, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 2>) mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 2>) mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 2>) mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 2>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 2, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 5>) mlx::red::row_reduce_small<half, half, mlx::red::Max<half>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 5>) mlx::red::row_reduce_looped<half, half, mlx::red::Max<half>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 5>) mlx::red::col_reduce_small<half, half, mlx::red::Max<half>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 5>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Max<half>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 5, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Max<half>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Max<half>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_bfloat16")]] [[kernel]] decltype(mlx::red::all_reduce<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int64_t>) mlx::red::all_reduce<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, size_t>) mlx::red::row_reduce_simple<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Max<bfloat16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_max_float32")]] [[kernel]] decltype(mlx::red::all_reduce<float, float, mlx::red::Max<float>, int64_t>) mlx::red::all_reduce<float, float, mlx::red::Max<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<float, float, mlx::red::Max<float>, size_t>) mlx::red::row_reduce_simple<float, float, mlx::red::Max<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 1>) mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 1>) mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 1>) mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 1>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 2>) mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 2>) mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 2>) mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 2>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 5>) mlx::red::row_reduce_small<float, float, mlx::red::Max<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_max_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 5>) mlx::red::row_reduce_looped<float, float, mlx::red::Max<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 5>) mlx::red::col_reduce_small<float, float, mlx::red::Max<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 5>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Max<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Max<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_max_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Max<float>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_int8")]] [[kernel]] decltype(mlx::red::all_reduce<int8_t, int8_t, mlx::red::Min<int8_t>, int64_t>) mlx::red::all_reduce<int8_t, int8_t, mlx::red::Min<int8_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int8_t, int8_t, mlx::red::Min<int8_t>, size_t>) mlx::red::row_reduce_simple<int8_t, int8_t, mlx::red::Min<int8_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>) mlx::red::row_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_int8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>) mlx::red::row_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>) mlx::red::col_reduce_small<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>) mlx::red::col_reduce_longcolumn<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_int8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int8_t, int8_t, mlx::red::Min<int8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_int16")]] [[kernel]] decltype(mlx::red::all_reduce<int16_t, int16_t, mlx::red::Min<int16_t>, int64_t>) mlx::red::all_reduce<int16_t, int16_t, mlx::red::Min<int16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int16_t, int16_t, mlx::red::Min<int16_t>, size_t>) mlx::red::row_reduce_simple<int16_t, int16_t, mlx::red::Min<int16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>) mlx::red::row_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_int16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>) mlx::red::row_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>) mlx::red::col_reduce_small<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>) mlx::red::col_reduce_longcolumn<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_int16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int16_t, int16_t, mlx::red::Min<int16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_int32")]] [[kernel]] decltype(mlx::red::all_reduce<int32_t, int32_t, mlx::red::Min<int32_t>, int64_t>) mlx::red::all_reduce<int32_t, int32_t, mlx::red::Min<int32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Min<int32_t>, size_t>) mlx::red::row_reduce_simple<int32_t, int32_t, mlx::red::Min<int32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>) mlx::red::row_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_int32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>) mlx::red::row_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>) mlx::red::col_reduce_small<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>) mlx::red::col_reduce_longcolumn<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_int32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int32_t, int32_t, mlx::red::Min<int32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_int64")]] [[kernel]] decltype(mlx::red::all_reduce<int64_t, int64_t, mlx::red::Min<int64_t>, int64_t>) mlx::red::all_reduce<int64_t, int64_t, mlx::red::Min<int64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Min<int64_t>, size_t>) mlx::red::row_reduce_simple<int64_t, int64_t, mlx::red::Min<int64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>) mlx::red::row_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_int64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>) mlx::red::row_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>) mlx::red::col_reduce_small<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>) mlx::red::col_reduce_longcolumn<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_int64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<int64_t, int64_t, mlx::red::Min<int64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_uint8")]] [[kernel]] decltype(mlx::red::all_reduce<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int64_t>) mlx::red::all_reduce<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint8_t, uint8_t, mlx::red::Min<uint8_t>, size_t>) mlx::red::row_reduce_simple<uint8_t, uint8_t, mlx::red::Min<uint8_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>) mlx::red::row_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_uint8")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>) mlx::red::row_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>) mlx::red::col_reduce_small<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_uint8")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint8_t, uint8_t, mlx::red::Min<uint8_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_uint16")]] [[kernel]] decltype(mlx::red::all_reduce<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int64_t>) mlx::red::all_reduce<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint16_t, uint16_t, mlx::red::Min<uint16_t>, size_t>) mlx::red::row_reduce_simple<uint16_t, uint16_t, mlx::red::Min<uint16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>) mlx::red::row_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_uint16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>) mlx::red::row_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>) mlx::red::col_reduce_small<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_uint16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint16_t, uint16_t, mlx::red::Min<uint16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_uint32")]] [[kernel]] decltype(mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int64_t>) mlx::red::all_reduce<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Min<uint32_t>, size_t>) mlx::red::row_reduce_simple<uint32_t, uint32_t, mlx::red::Min<uint32_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>) mlx::red::row_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_uint32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>) mlx::red::row_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>) mlx::red::col_reduce_small<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_uint32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint32_t, uint32_t, mlx::red::Min<uint32_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_uint64")]] [[kernel]] decltype(mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int64_t>) mlx::red::all_reduce<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Min<uint64_t>, size_t>) mlx::red::row_reduce_simple<uint64_t, uint64_t, mlx::red::Min<uint64_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>) mlx::red::row_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_uint64")]] [[kernel]] decltype(mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>) mlx::red::row_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>) mlx::red::col_reduce_small<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>) mlx::red::col_reduce_longcolumn<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_uint64")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<uint64_t, uint64_t, mlx::red::Min<uint64_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_float16")]] [[kernel]] decltype(mlx::red::all_reduce<half, half, mlx::red::Min<half>, int64_t>) mlx::red::all_reduce<half, half, mlx::red::Min<half>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<half, half, mlx::red::Min<half>, size_t>) mlx::red::row_reduce_simple<half, half, mlx::red::Min<half>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 1>) mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 1>) mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 1>) mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 1>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 1, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 2>) mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 2>) mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 2>) mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 2>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 2, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 5>) mlx::red::row_reduce_small<half, half, mlx::red::Min<half>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_float16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 5>) mlx::red::row_reduce_looped<half, half, mlx::red::Min<half>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 5>) mlx::red::col_reduce_small<half, half, mlx::red::Min<half>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 5>) mlx::red::col_reduce_longcolumn<half, half, mlx::red::Min<half>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 5, 32, 32>) mlx::red::col_reduce_looped<half, half, mlx::red::Min<half>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_float16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<half, half, mlx::red::Min<half>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_bfloat16")]] [[kernel]] decltype(mlx::red::all_reduce<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int64_t>) mlx::red::all_reduce<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_simple<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, size_t>) mlx::red::row_reduce_simple<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>) mlx::red::row_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_bfloat16")]] [[kernel]] decltype(mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>) mlx::red::row_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>) mlx::red::col_reduce_small<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>) mlx::red::col_reduce_longcolumn<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5, 32, 32>) mlx::red::col_reduce_looped<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_bfloat16")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<bfloat16_t, bfloat16_t, mlx::red::Min<bfloat16_t>, int, 5, 32, 32>;
template [[host_name("nuvec_all_reduce_min_float32")]] [[kernel]] decltype(mlx::red::all_reduce<float, float, mlx::red::Min<float>, int64_t>) mlx::red::all_reduce<float, float, mlx::red::Min<float>, int64_t>;
template [[host_name("nuvec_row_reduce_simple_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_simple<float, float, mlx::red::Min<float>, size_t>) mlx::red::row_reduce_simple<float, float, mlx::red::Min<float>, size_t>;
template [[host_name("nuvec_row_reduce_small_1_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 1>) mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 1>;
template [[host_name("nuvec_row_reduce_looped_1_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 1>) mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 1>;
template [[host_name("nuvec_col_reduce_small_1_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 1>) mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 1>;
template [[host_name("nuvec_col_reduce_longcolumn_1_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 1>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 1>;
template [[host_name("nuvec_col_reduce_looped_1_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 1, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 1, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_1_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 1, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 1, 32, 32>;
template [[host_name("nuvec_row_reduce_small_2_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 2>) mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 2>;
template [[host_name("nuvec_row_reduce_looped_2_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 2>) mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 2>;
template [[host_name("nuvec_col_reduce_small_2_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 2>) mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 2>;
template [[host_name("nuvec_col_reduce_longcolumn_2_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 2>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 2>;
template [[host_name("nuvec_col_reduce_looped_2_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 2, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 2, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_2_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 2, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 2, 32, 32>;
template [[host_name("nuvec_row_reduce_small_5_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 5>) mlx::red::row_reduce_small<float, float, mlx::red::Min<float>, int, 5>;
template [[host_name("nuvec_row_reduce_looped_5_min_float32")]] [[kernel]] decltype(mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 5>) mlx::red::row_reduce_looped<float, float, mlx::red::Min<float>, int, 5>;
template [[host_name("nuvec_col_reduce_small_5_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 5>) mlx::red::col_reduce_small<float, float, mlx::red::Min<float>, int, 5>;
template [[host_name("nuvec_col_reduce_longcolumn_5_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 5>) mlx::red::col_reduce_longcolumn<float, float, mlx::red::Min<float>, int, 5>;
template [[host_name("nuvec_col_reduce_looped_5_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 5, 32, 32>) mlx::red::col_reduce_looped<float, float, mlx::red::Min<float>, int, 5, 32, 32>;
template [[host_name("nuvec_col_reduce_2pass_5_32_32_min_float32")]] [[kernel]] decltype(mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 5, 32, 32>) mlx::red::col_reduce_2pass<float, float, mlx::red::Min<float>, int, 5, 32, 32>;
