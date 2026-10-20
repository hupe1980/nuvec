// SPDX-License-Identifier: MIT
//
// Rotary position embeddings, ported from MLX v0.32.4 (mlx/backend/metal/kernels/rope.metal) into `mlx::rope`.
// The function constants are renamed `rope_forward`, `rope_traditional` and `rope_transpose` (fast.h).
// `norm_rope` (normalisation, rotation and a strided write in one kernel) is nuvec's own.
//
// Copyright © 2023-2026 Apple Inc. Licensed under the MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

#include "fast.h"

namespace mlx {
namespace rope {

template <typename T>
void rope_single_impl(
    const device T* in,
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t& stride,
    uint2 pos,
    uint2 grid) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  uint index_1, index_2;
  if (rope_traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + grid.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (rope_forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T>
[[kernel]] void rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T>
[[kernel]] void rope_single_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, typename IdxT, int N = 4>
void rope_impl(
    const device T* in,
    device T* out,
    const device int* offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    uint3 pos,
    uint3 grid) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);
  auto mat_idx = batch_idx * n_head + head_idx;

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  // Compute the input and output indices
  IdxT in_index_1;
  if (rope_transpose) {
    IdxT batch_stride = grid.y * IdxT(strides[1]);
    in_index_1 =
        batch_idx * batch_stride + pos.y * strides[1] + head_idx * strides[0];
  } else {
    in_index_1 = pos.y * IdxT(strides[1]) + mat_idx * IdxT(strides[0]);
  }
  IdxT in_index_2;
  IdxT out_index_1 =
      pos.y * IdxT(out_strides[1]) + mat_idx * IdxT(out_strides[0]);
  IdxT out_index_2;
  if (rope_traditional) {
    out_index_1 += 2 * pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + 1;
    in_index_1 += 2 * pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + IdxT(strides[2]);
  } else {
    out_index_1 += pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + grid.x * IdxT(out_strides[2]);
    in_index_1 += pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + grid.x * IdxT(strides[2]);
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (rope_forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += IdxT(strides[0]);
    in_index_2 += IdxT(strides[0]);
    out_index_1 += IdxT(out_strides[0]);
    out_index_2 += IdxT(out_strides[0]);
  }
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

} // namespace rope
} // namespace mlx

template [[host_name("nuvec_rope_single_float32")]] [[kernel]] decltype(mlx::rope::rope_single<float>) mlx::rope::rope_single<float>;
template [[host_name("nuvec_rope_single_freqs_float32")]] [[kernel]] decltype(mlx::rope::rope_single_freqs<float>) mlx::rope::rope_single_freqs<float>;
template [[host_name("nuvec_rope_float32")]] [[kernel]] decltype(mlx::rope::rope<float, int32_t>) mlx::rope::rope<float, int32_t>;
template [[host_name("nuvec_rope_freqs_float32")]] [[kernel]] decltype(mlx::rope::rope_freqs<float, int32_t>) mlx::rope::rope_freqs<float, int32_t>;
template [[host_name("nuvec_rope_large_float32")]] [[kernel]] decltype(mlx::rope::rope<float, int64_t>) mlx::rope::rope<float, int64_t>;
template [[host_name("nuvec_rope_freqs_large_float32")]] [[kernel]] decltype(mlx::rope::rope_freqs<float, int64_t>) mlx::rope::rope_freqs<float, int64_t>;
template [[host_name("nuvec_rope_single_float16")]] [[kernel]] decltype(mlx::rope::rope_single<half>) mlx::rope::rope_single<half>;
template [[host_name("nuvec_rope_single_freqs_float16")]] [[kernel]] decltype(mlx::rope::rope_single_freqs<half>) mlx::rope::rope_single_freqs<half>;
template [[host_name("nuvec_rope_float16")]] [[kernel]] decltype(mlx::rope::rope<half, int32_t>) mlx::rope::rope<half, int32_t>;
template [[host_name("nuvec_rope_freqs_float16")]] [[kernel]] decltype(mlx::rope::rope_freqs<half, int32_t>) mlx::rope::rope_freqs<half, int32_t>;
template [[host_name("nuvec_rope_large_float16")]] [[kernel]] decltype(mlx::rope::rope<half, int64_t>) mlx::rope::rope<half, int64_t>;
template [[host_name("nuvec_rope_freqs_large_float16")]] [[kernel]] decltype(mlx::rope::rope_freqs<half, int64_t>) mlx::rope::rope_freqs<half, int64_t>;
template [[host_name("nuvec_rope_single_bfloat16")]] [[kernel]] decltype(mlx::rope::rope_single<bfloat16_t>) mlx::rope::rope_single<bfloat16_t>;
template [[host_name("nuvec_rope_single_freqs_bfloat16")]] [[kernel]] decltype(mlx::rope::rope_single_freqs<bfloat16_t>) mlx::rope::rope_single_freqs<bfloat16_t>;
template [[host_name("nuvec_rope_bfloat16")]] [[kernel]] decltype(mlx::rope::rope<bfloat16_t, int32_t>) mlx::rope::rope<bfloat16_t, int32_t>;
template [[host_name("nuvec_rope_freqs_bfloat16")]] [[kernel]] decltype(mlx::rope::rope_freqs<bfloat16_t, int32_t>) mlx::rope::rope_freqs<bfloat16_t, int32_t>;
template [[host_name("nuvec_rope_large_bfloat16")]] [[kernel]] decltype(mlx::rope::rope<bfloat16_t, int64_t>) mlx::rope::rope<bfloat16_t, int64_t>;
template [[host_name("nuvec_rope_freqs_large_bfloat16")]] [[kernel]] decltype(mlx::rope::rope_freqs<bfloat16_t, int64_t>) mlx::rope::rope_freqs<bfloat16_t, int64_t>;

// nuvec's own: per-row RMS normalisation (optional, as `rms_single_row`: the same reads
// per thread, so the same sums) and the rotation (as `rope_single`/`rope`: the same
// angles and rounding) in one pass, written at `write_offset + l` of the output's rows,
// for instance into a key cache. One threadgroup per `[b, h, l]` row of at most
// `NORM_ROPE_MAX` features; `x` is read through its strides (heads may be transposed).
static constant constexpr int NORM_ROPE_MAX = 1024;

template <typename T, bool NORM, bool FREQS>
[[kernel]] void norm_rope(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant int& offset [[buffer(4)]],
    constant int& write_offset [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int64_t* x_strides [[buffer(7)]],
    constant int64_t* out_strides [[buffer(8)]],
    constant int* sizes [[buffer(9)]],
    constant float& base [[buffer(10)]],
    const device float* freqs [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = RMS_N_READS;
  constexpr int SIMD_SIZE = 32;
  const int heads = sizes[0], length = sizes[1], d = sizes[2], dims = sizes[3];
  const uint axis_size = d;
  const int l = gid % length;
  const int bh = gid / length;
  const int h = bh % heads, b = bh / heads;
  x += b * x_strides[0] + h * x_strides[1] + l * x_strides[2] + lid * N_READS;
  out += b * out_strides[0] + h * out_strides[1] +
      int64_t(write_offset + l) * out_strides[2];

  threadgroup T row[NORM_ROPE_MAX];
  float thread_x[N_READS];
  float acc = 0;
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
    acc += thread_x[i] * thread_x[i];
  }
  if (NORM) {
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      uint j = lid * N_READS + i;
      if (j < axis_size) {
        row[j] = w[j] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      uint j = lid * N_READS + i;
      if (j < axis_size) {
        row[j] = static_cast<T>(thread_x[i]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int half_dims = dims / 2;
  const float L = scale * static_cast<float>(offset + l);
  for (int i = 0; i < N_READS; i++) {
    int j = lid * N_READS + i;
    if (j >= d) {
      break;
    }
    if (j >= dims) {
      out[j] = row[j];
      continue;
    }
    int p;
    bool second;
    float x1, x2;
    if (rope_traditional) {
      p = j / 2;
      second = j & 1;
      x1 = static_cast<float>(row[2 * p]);
      x2 = static_cast<float>(row[2 * p + 1]);
    } else {
      second = j >= half_dims;
      p = second ? j - half_dims : j;
      x1 = static_cast<float>(row[p]);
      x2 = static_cast<float>(row[p + half_dims]);
    }
    float inv_freq;
    if (FREQS) {
      inv_freq = 1.0 / (freqs[p]);
    } else {
      float dp = static_cast<float>(p) / static_cast<float>(half_dims);
      inv_freq = metal::exp2(-dp * base);
    }
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    out[j] = static_cast<T>(
        second ? x1 * sintheta + x2 * costheta : x1 * costheta - x2 * sintheta);
  }
}

#define instantiate_norm_rope(tname, type)                                         \
  template [[host_name("nuvec_norm_rope_" #tname)]] [[kernel]]                     \
  decltype(norm_rope<type, true, false>) norm_rope<type, true, false>;             \
  template [[host_name("nuvec_norm_rope_freqs_" #tname)]] [[kernel]]               \
  decltype(norm_rope<type, true, true>) norm_rope<type, true, true>;               \
  template [[host_name("nuvec_plain_rope_" #tname)]] [[kernel]]                    \
  decltype(norm_rope<type, false, false>) norm_rope<type, false, false>;           \
  template [[host_name("nuvec_plain_rope_freqs_" #tname)]] [[kernel]]              \
  decltype(norm_rope<type, false, true>) norm_rope<type, false, true>;

instantiate_norm_rope(float32, float)
instantiate_norm_rope(float16, half)
instantiate_norm_rope(bfloat16, bfloat16_t)
