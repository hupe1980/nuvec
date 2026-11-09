//! Launching the Tile kernels (`nuvec_kernels::cuda::tile`) on nuvec's managed buffers,
//! borrowed without a copy, with their tile-size heuristics.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cutile::tensor::{IntoPartition, Tensor};
use cutile::tile_kernel::TileKernel;
use nuvec_kernels::cuda::ptx::DType;
use nuvec_kernels::cuda::tile::kernels;

use super::device::Device;
use super::error::{CudaError, Result};
use super::memory::Buffer;

/// How f32 products use the tensor cores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32Mode {
    /// 3×TF32: f32 accuracy (the CPU oracle's tolerance) on the tensor cores. The default.
    Exact,
    /// One TF32 product: 10-bit mantissa inputs, as MLX's default (`MLX_ENABLE_TF32`).
    /// `NUVEC_CUDA_TF32=1`.
    Tf32,
}

impl F32Mode {
    /// From `NUVEC_CUDA_TF32` (read once).
    pub fn current() -> Self {
        static MODE: std::sync::OnceLock<F32Mode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("NUVEC_CUDA_TF32").as_deref() {
            Ok("1") => F32Mode::Tf32,
            _ => F32Mode::Exact,
        })
    }
}

/// Tile sizes `(BM, BN, BK)` for a GEMM of `m × n` with elements of `dtype` on compute
/// capability `cc` (dispatch heuristics ship with the kernel), or
/// `NUVEC_CUDA_GEMM_TILES=bm,bn,bk` for measurements.
pub fn gemm_tiles(dtype: DType, m: usize, n: usize, _cc: (u32, u32)) -> (usize, usize, usize) {
    if let Ok(v) = std::env::var("NUVEC_CUDA_GEMM_TILES") {
        let t: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if let [bm, bn, bk] = t[..] {
            return (bm, bn, bk);
        }
    }
    // On sm_86 (A40), 128 × 128 × 32 reaches 33 TFLOP/s at 4096 in f16; 128 × 256 and
    // larger collapse to 1–3 (registers and shared memory). f32 has its own sweep.
    let _ = (m, n);
    match dtype {
        DType::F32 => (64, 64, 32),
        _ => (128, 128, 32),
    }
}

/// Borrows `buffer` (offset 0) as a contiguous `rows × cols` tensor of `T`.
///
/// # Safety
/// The buffer must hold `rows · cols` elements of `T` and outlive the returned tensor's
/// GPU use (the caller synchronises the stream before dropping it).
unsafe fn borrow<T: cutile::DType>(buffer: &Buffer, rows: usize, cols: usize) -> Tensor<T> {
    let shape = vec![rows as i32, cols as i32];
    let strides = vec![cols as i32, 1];
    // SAFETY: forwarded to the caller.
    unsafe { Tensor::borrow_raw_parts(buffer.device_ptr(), 0, shape, strides) }
}

fn tile_error(e: impl std::fmt::Display) -> CudaError {
    CudaError::Tile(e.to_string())
}

/// `out = x · y` for row-major `x` `[m, k]`, `y` `[k, n]` and `out` `[m, n]`, all of
/// `dtype`, with f32 accumulation. Waits for the result.
pub fn gemm(
    device: &'static Device,
    dtype: DType,
    out: &mut Buffer,
    x: &Buffer,
    y: &Buffer,
    (m, n, k): (usize, usize, usize),
) -> Result<()> {
    if [m, n, k].iter().any(|&d| d == 0 || d > i32::MAX as usize) {
        return Err(CudaError::InvalidArgument(format!(
            "gemm of {m}×{k} by {k}×{n}"
        )));
    }
    let (bm, bn, bk) = gemm_tiles(dtype, m, n, device.info().compute_capability);
    let name = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
    };
    let aligned = m.is_multiple_of(bm) && n.is_multiple_of(bn) && k.is_multiple_of(bk);
    let tiles = [bm.to_string(), bn.to_string(), bk.to_string()];
    let kk = k as i32;
    device
        .device
        .bind_to_thread()
        .map_err(super::error::driver("cuCtxSetCurrent"))?;
    macro_rules! borrowed {
        ($t:ty) => {
            // SAFETY: each buffer holds its matrix; every launch below is synchronised
            // before the borrows end (`sync_on`).
            unsafe {
                (
                    borrow::<$t>(out, m, n),
                    Arc::new(borrow::<$t>(x, m, k)),
                    Arc::new(borrow::<$t>(y, k, n)),
                )
            }
        };
    }
    macro_rules! run {
        ($t:ty) => {{
            let (z, a, b) = borrowed!($t);
            if aligned {
                let mut g = vec![name.to_string()];
                g.extend(tiles.clone());
                // SAFETY: the shapes are multiples of the tiles, so no access leaves them.
                let launch = unsafe { kernels::gemm_fast(z.partition([bm, bn]), a, b, kk) };
                launch
                    .generics(g)
                    .sync_on(&device.stream)
                    .map_err(tile_error)?;
            } else {
                let mut g = vec![name.to_string()];
                g.extend(tiles.clone());
                g.push(k.to_string());
                let launch = kernels::gemm(z.partition([bm, bn]), a, b).generics(g);
                launch.sync_on(&device.stream).map_err(tile_error)?;
            }
        }};
    }
    match dtype {
        DType::F32 if aligned => {
            let (z, a, b) = borrowed!(f32);
            let g = tiles.to_vec();
            let z = z.partition([bm, bn]);
            match F32Mode::current() {
                // SAFETY: as above, aligned shapes.
                F32Mode::Exact => unsafe { kernels::gemm_3xtf32(z, a, b, kk) }
                    .generics(g)
                    .sync_on(&device.stream)
                    .map(|_| ()),
                // SAFETY: as above, aligned shapes.
                F32Mode::Tf32 => unsafe { kernels::gemm_tf32(z, a, b, kk) }
                    .generics(g)
                    .sync_on(&device.stream)
                    .map(|_| ()),
            }
            .map_err(tile_error)?;
        }
        DType::F32 => run!(f32),
        DType::F16 => run!(half::f16),
        DType::BF16 => run!(half::bf16),
    }
    Ok(())
}
