//! Launching the generated PTX kernels on the shared device's stream.

use std::ffi::c_void;

use super::device::Device;
use super::error::{CudaError, Result};
use super::memory::Buffer;
use nuvec_kernels::cuda::ptx::{self, DType, MAX_DIMS, Op};

const BLOCK: u32 = 256;

/// An input view: a buffer, its first element's offset (elements) and, for strided
/// kernels, its element strides per dimension.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    /// The buffer.
    pub buffer: &'a Buffer,
    /// The first element, in elements.
    pub offset: usize,
    /// Element strides, one per dimension of the output (strided kernels only).
    pub strides: &'a [i32],
}

fn count(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| CudaError::InvalidArgument(format!("{n} elements exceed 2^32")))
}

/// `out[i] = op(inputs[k][i]…)` over `n` contiguous elements (inputs at their offsets).
pub fn elementwise(
    device: &'static Device,
    op: Op,
    dtype: DType,
    out: &mut Buffer,
    inputs: &[View<'_>],
    n: usize,
) -> Result<()> {
    launch(device, op, dtype, out, inputs, n, None)
}

/// `out[i] = op(inputs[k][index(i)]…)` with `out` contiguous in `shape` and each input
/// addressed through its own strides (broadcast dimensions have stride 0).
pub fn elementwise_strided(
    device: &'static Device,
    op: Op,
    dtype: DType,
    out: &mut Buffer,
    inputs: &[View<'_>],
    shape: &[usize],
) -> Result<()> {
    if shape.len() > MAX_DIMS || inputs.iter().any(|v| v.strides.len() != shape.len()) {
        return Err(CudaError::InvalidArgument(format!(
            "strided element-wise kernels take up to {MAX_DIMS} dimensions, one stride each"
        )));
    }
    launch(
        device,
        op,
        dtype,
        out,
        inputs,
        shape.iter().product(),
        Some(shape),
    )
}

fn launch(
    device: &'static Device,
    op: Op,
    dtype: DType,
    out: &mut Buffer,
    inputs: &[View<'_>],
    n: usize,
    shape: Option<&[usize]>,
) -> Result<()> {
    if inputs.len() != op.arity() {
        return Err(CudaError::InvalidArgument(format!(
            "{op:?} takes {} inputs",
            op.arity()
        )));
    }
    let n32 = count(n)?;
    if n == 0 {
        return Ok(());
    }
    let kernel = ptx::elementwise(op, dtype, shape.map(<[usize]>::len));
    let f = device.function(&kernel.ptx, &kernel.name)?;
    let size = dtype.size() as u64;
    let mut out_ptr = out.device_ptr();
    let mut in_ptrs: Vec<u64> = inputs
        .iter()
        .map(|v| v.buffer.device_ptr() + v.offset as u64 * size)
        .collect();
    let mut n_arg = n32;
    let mut dims = [0u32; MAX_DIMS];
    let mut strides = vec![[0i32; MAX_DIMS]; inputs.len()];
    if let Some(shape) = shape {
        for (d, &s) in shape.iter().enumerate() {
            dims[d] = count(s)?;
        }
        for (k, v) in inputs.iter().enumerate() {
            strides[k][..v.strides.len()].copy_from_slice(v.strides);
        }
    }
    let mut args: Vec<*mut c_void> = vec![(&raw mut out_ptr).cast()];
    args.extend(in_ptrs.iter_mut().map(|p| (p as *mut u64).cast::<c_void>()));
    args.push((&raw mut n_arg).cast());
    if shape.is_some() {
        args.push((&raw mut dims).cast());
        args.extend(
            strides
                .iter_mut()
                .map(|s| (s as *mut [i32; MAX_DIMS]).cast::<c_void>()),
        );
    }
    device.launch(&f, n32.div_ceil(BLOCK), BLOCK, 0, &mut args)
}

/// `out[r] = Σ input[r, :]` over `rows` contiguous rows of `len` elements.
pub fn row_sum(
    device: &'static Device,
    dtype: DType,
    out: &mut Buffer,
    input: &Buffer,
    len: usize,
    rows: usize,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    let kernel = ptx::row_sum(dtype);
    let f = device.function(&kernel.ptx, &kernel.name)?;
    let (mut o, mut i, mut l, mut r) = (
        out.device_ptr(),
        input.device_ptr(),
        count(len)?,
        count(rows)?,
    );
    let mut args: Vec<*mut c_void> = vec![
        (&raw mut o).cast(),
        (&raw mut i).cast(),
        (&raw mut l).cast(),
        (&raw mut r).cast(),
    ];
    device.launch(&f, r, ptx::ROW_THREADS, 0, &mut args)
}
