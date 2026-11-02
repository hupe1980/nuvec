// nuvec's own: the glue around a mixture of experts' products, each kernel the arithmetic
// of its composition in the same order (so the same bits).
//
// `moe_route`: per row of `E` probabilities, the `k` largest in the stable ascending order
// of the sort kernel (`argsort`, whose last `k` are `argpartition`'s), their values and,
// with `normalize`, those divided by their sum (`row_reduce_small`'s sum: in the
// accumulator type, in order, rounded once; then the element-wise divide). One
// threadgroup per row, one thread per expert: an expert's position in the sorted row is
// the number of experts sorting before it.
//
// `weighted_sum`: `sum(x * w[..., None], -2)` over `[R, k, D]` as `col_reduce_small` sums
// it: `min(k, 8)` partial sums of every `min(k, 8)`-th product (each rounded to `T`, as the
// element-wise multiply), added in order in the accumulator type, rounded once (one
// column is a row reduction: in order).

#include "fast.h"

// The sort kernel's `LessThan`: NaN sorts last.
template <typename T>
inline bool route_less(T a, T b) {
  if constexpr (metal::is_floating_point_v<T>) {
    bool an = metal::isnan(a);
    bool bn = metal::isnan(b);
    if (an | bn) {
      return (!an) & bn;
    }
  }
  return a < b;
}

static constant constexpr int MOE_ROUTE_MAX = 1024;

template <typename T, typename U>
[[kernel]] void moe_route(
    const device T* probs [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device T* scores [[buffer(2)]],
    constant int& experts [[buffer(3)]],
    constant int& k [[buffer(4)]],
    constant int& normalize [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint e [[thread_position_in_threadgroup]]) {
  threadgroup T p[MOE_ROUTE_MAX];
  threadgroup T top[MOE_ROUTE_MAX];
  threadgroup T total[1];
  const int n = experts;
  probs += size_t(row) * n;
  indices += size_t(row) * k;
  scores += size_t(row) * k;
  if (int(e) < n) {
    p[e] = probs[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(e) < n) {
    const T v = p[e];
    int rank = 0;
    for (int j = 0; j < n; j++) {
      const T u = p[j];
      rank += (route_less(u, v) || (!route_less(v, u) && j < int(e))) ? 1 : 0;
    }
    const int slot = rank - (n - k);
    if (slot >= 0) {
      indices[slot] = e;
      top[slot] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (normalize) {
    if (e == 0) {
      U sum = 0;
      for (int j = 0; j < k; j++) {
        sum = static_cast<U>(top[j]) + sum;
      }
      total[0] = static_cast<T>(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(e) < k) {
      scores[e] = top[e] / total[0];
    }
  } else if (int(e) < k) {
    scores[e] = top[e];
  }
}

template <typename T, typename U>
[[kernel]] void weighted_sum(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int& k [[buffer(3)]],
    constant int& d [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const size_t r = pos.y;
  x += r * size_t(k) * d + pos.x;
  w += r * size_t(k);
  // One column (`D = 1`): the reduced axis is contiguous, a row reduction, in order.
  const int lanes = d == 1 ? 1 : metal::min(k, 8);
  U parts[8];
  for (int y = 0; y < lanes; y++) {
    U t = 0;
    for (int j = y; j < k; j += lanes) {
      const T product = x[size_t(j) * d] * w[j];
      t = static_cast<U>(product) + t;
    }
    parts[y] = t;
  }
  U sum = parts[0];
  for (int y = 1; y < lanes; y++) {
    sum = parts[y] + sum;
  }
  out[r * d + pos.x] = static_cast<T>(sum);
}

#define instantiate_moe(tname, type, acc)                                    \
  template [[host_name("nuvec_moe_route_" #tname)]] [[kernel]]               \
  decltype(moe_route<type, acc>) moe_route<type, acc>;                       \
  template [[host_name("nuvec_weighted_sum_" #tname)]] [[kernel]]            \
  decltype(weighted_sum<type, acc>) weighted_sum<type, acc>;

instantiate_moe(float32, float, float)
instantiate_moe(float16, half, float)
instantiate_moe(bfloat16, bfloat16_t, float)
