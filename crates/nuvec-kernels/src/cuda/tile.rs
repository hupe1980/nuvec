//! Tile kernels (cutile): Rust kernels over tiles, JIT-compiled at first launch through the
//! CUDA toolkit's `tileiras` (CUDA 13.2 or later) and cached. `nuvec_gpu::cuda` launches
//! them on nuvec's buffers.

// `#[cutile::module]` generates launch wrappers without documentation.
#![allow(missing_docs)]

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// `z = x · y` for row-major `x` `[M, K]` and `y` `[K, N]`: one `[BM, BN]` tile of `z`
    /// per block, accumulated in f32 over `⌈K / BK⌉` tiles of K (edge tiles are masked by
    /// the partitions), then rounded to `E`.
    #[cutile::entry()]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..((K + BK - 1) / BK) {
            let a: Tile<E, { [BM, BK] }> = part_x.load([pid.0, i]);
            let b: Tile<E, { [BK, BN] }> = part_y.load([i, pid.1]);
            acc = mma(a, b, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out);
    }

    /// [`gemm`] without bounds checks, for `M`, `N` and `K` multiples of the tile sizes
    /// (cutile's own GEMM benchmark form): every load is a whole tile.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_fast<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, -1] }>,
        y: &Tensor<E, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let a: Tile<E, { [BM, BK] }> = part_x.load([pid.0, i]);
            let b: Tile<E, { [BK, BN] }> = part_y.load([i, pid.1]);
            acc = mma(a, b, acc);
        }
        let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
        z.store(out);
    }

    /// An f32 GEMM on the tensor cores in TF32 (10-bit mantissa inputs, f32 accumulation):
    /// MLX's default for f32 products (`MLX_ENABLE_TF32`). Aligned shapes only.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_tf32<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let a: Tile<f32, { [BM, BK] }> = part_x.load([pid.0, i]);
            let b: Tile<f32, { [BK, BN] }> = part_y.load([i, pid.1]);
            let ah: Tile<tf32, { [BM, BK] }> = convert_tile(a);
            let bh: Tile<tf32, { [BK, BN] }> = convert_tile(b);
            acc = mma(ah, bh, acc);
        }
        z.store(acc);
    }

    /// An f32 GEMM to f32 accuracy on the tensor cores: each operand is split exactly into
    /// a high part of 11 significant bits and the remainder (`x = hi + lo`), both TF32, and
    /// `x·y ≈ hi·hi + hi·lo + lo·hi` (the `lo·lo` term is below f32 rounding). Three TF32
    /// products, smallest first, instead of SIMT FMAs. Aligned shapes only.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_3xtf32<const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        let split_a: Tile<f32, { [BM, BK] }> = constant(8193.0f32, const_shape![BM, BK]);
        let split_b: Tile<f32, { [BK, BN] }> = constant(8193.0f32, const_shape![BK, BN]);
        for i in 0i32..(k / BK) {
            let a: Tile<f32, { [BM, BK] }> = part_x.load([pid.0, i]);
            let b: Tile<f32, { [BK, BN] }> = part_y.load([i, pid.1]);
            // Veltkamp's split: `hi` keeps 11 significant bits (exact in TF32), `lo` the
            // rest. (Converting to `tf32` labels the bits for the tensor cores; it does not
            // round them, so the split is explicit.)
            let ca: Tile<f32, { [BM, BK] }> = a * split_a;
            let cb: Tile<f32, { [BK, BN] }> = b * split_b;
            let ah32: Tile<f32, { [BM, BK] }> = ca - (ca - a);
            let bh32: Tile<f32, { [BK, BN] }> = cb - (cb - b);
            let ah: Tile<tf32, { [BM, BK] }> = convert_tile(ah32);
            let bh: Tile<tf32, { [BK, BN] }> = convert_tile(bh32);
            let al: Tile<tf32, { [BM, BK] }> = convert_tile(a - ah32);
            let bl: Tile<tf32, { [BK, BN] }> = convert_tile(b - bh32);
            acc = mma(al, bh, acc);
            acc = mma(ah, bl, acc);
            acc = mma(ah, bh, acc);
        }
        z.store(acc);
    }
}
