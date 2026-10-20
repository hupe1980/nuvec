// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Random bits (nuvec's own), with MLX's layout (`RandomBits::eval_cpu`, mirrored by
// nuvec-core's `kernels::random_bits`): per key, words `i` and `half + odd + i` are the
// threefry2x32 hash of counter `(i, half + odd + i)`, and an odd word count takes its
// middle word from counter `(half, 0)`. One thread per counter.

#include <metal_stdlib>

using namespace metal;

static inline uint rotl(uint x, uint r) {
    return (x << r) | (x >> (32 - r));
}

// threefry2x32 with 20 rounds, as MLX's `threefry2x32_hash`.
static inline uint2 threefry2x32(uint2 key, uint2 count) {
    const uint rotations[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    const uint ks[3] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};
    count.x += ks[0];
    count.y += ks[1];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 4; ++j) {
            count.x += count.y;
            count.y = rotl(count.y, rotations[i % 2][j]) ^ count.x;
        }
        count.x += ks[(i + 1) % 3];
        count.y += ks[(i + 2) % 3] + uint(i + 1);
    }
    return count;
}

// Writes the bytes of `word` that fall inside the key's `bytes_per_key`.
static inline void put(device uchar* out, uint word_index, uint word, uint bytes_per_key) {
    uint start = word_index * 4;
    for (uint b = 0; b < 4 && start + b < bytes_per_key; ++b) {
        out[start + b] = uchar(word >> (8 * b));
    }
}

// Grid: (half + odd, number of keys).
[[kernel]] void nuvec_random_bits(
    const device uint* keys [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& bytes_per_key [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
    const uint words = (bytes_per_key + 3) / 4;
    const uint half_words = words / 2, odd = words % 2;
    const uint2 key = uint2(keys[2 * pos.y], keys[2 * pos.y + 1]);
    device uchar* dst = out + size_t(pos.y) * bytes_per_key;
    const uint i = pos.x;
    if (i < half_words) {
        uint2 r = threefry2x32(key, uint2(i, half_words + odd + i));
        put(dst, i, r.x, bytes_per_key);
        put(dst, half_words + odd + i, r.y, bytes_per_key);
    } else if (odd == 1 && i == half_words) {
        put(dst, half_words, threefry2x32(key, uint2(half_words, 0)).x, bytes_per_key);
    }
}
