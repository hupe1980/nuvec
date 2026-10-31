//! `view` (MLX's `mx.view`): bit reinterpretation against a byte-level reference over random
//! shapes, layouts and offsets, its errors, `vmap` and derivatives, on the CPU and the GPU.

mod common;

use common::Rng;
use nuvec_core::*;

/// The array's bytes in row-major order.
fn bytes(a: &Array) -> Vec<u8> {
    match a.dtype().size() {
        1 => a.astype(DType::U8).to_vec::<u8>().unwrap(),
        2 => a
            .view(DType::U16)
            .to_vec::<u16>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        4 => a
            .view(DType::U32)
            .to_vec::<u32>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => a
            .view(DType::U64)
            .to_vec::<u64>()
            .unwrap()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

/// Random bytes as an array of `dtype` with `shape` (row-contiguous).
fn random_bytes(rng: &mut Rng, shape: &[i32], dtype: DType) -> Array {
    let n = shape.iter().product::<i32>() as usize * dtype.size();
    let raw: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
    let mut s = shape.to_vec();
    *s.last_mut().unwrap() *= dtype.size() as i32;
    Array::from_slice(&raw, &s[..]).view(dtype)
}

fn devices() -> Vec<Device> {
    let mut d = vec![Device::cpu()];
    if common::require_gpu().is_some() {
        d.push(Device::gpu(0));
    }
    d
}

#[test]
fn views_reinterpret_bytes_over_layouts() {
    let mut rng = Rng::new(0x71e);
    let pairs = [
        (DType::U8, DType::U32),
        (DType::U32, DType::U8),
        (DType::U8, DType::U16),
        (DType::F32, DType::U32),
        (DType::BF16, DType::U16),
        (DType::U16, DType::F16),
        (DType::U8, DType::U64),
    ];
    for device in devices() {
        with_stream(device, || {
            for (from, to) in pairs {
                for case in 0..5 {
                    let unit = (to.size() / from.size()).max(1) as i32;
                    let rows = 1 + rng.below(4) as i32;
                    let cols = unit * (1 + rng.below(6) as i32);
                    let x = match case {
                        // A transposed (column-major) matrix: copied first.
                        0 => random_bytes(&mut rng, &[cols, rows], from).t(),
                        // An offset slice: shared when the offset divides, else copied.
                        1 => random_bytes(&mut rng, &[rows + 1, cols], from).i((1.., ..)),
                        2 => random_bytes(&mut rng, &[1, cols + unit], from).i((.., unit..)),
                        3 => random_bytes(&mut rng, &[2, rows, cols], from),
                        _ => random_bytes(&mut rng, &[rows, cols], from),
                    };
                    let y = x.view(to);
                    let mut shape = x.shape().to_vec();
                    let last = shape.len() - 1;
                    shape[last] = shape[last] * from.size() as i32 / to.size() as i32;
                    assert_eq!(y.shape(), &shape[..], "{device:?} {from}→{to} case {case}");
                    assert_eq!(bytes(&y), bytes(&x), "{device:?} {from}→{to} case {case}");
                }
            }
            // Equal sizes keep the layout: a transposed float32 matrix viewed as uint32.
            let x = random_bytes(&mut rng, &[3, 5], DType::F32).t();
            let y = x.view(DType::U32);
            assert_eq!(y.shape(), [5, 3]);
            assert_eq!(bytes(&y), bytes(&x));
        });
    }
}

#[test]
fn view_errors_vmap_and_derivatives() {
    let x = Array::from_slice(&[1u8, 2, 3, 4, 5, 6], [2, 3]);
    assert!(
        x.view(DType::U32).check().is_err(),
        "last axis does not divide"
    );
    assert!(
        Array::from(1u8).view(DType::U32).check().is_err(),
        "scalar size change"
    );
    assert!(
        Array::from(1.0f32).view(DType::U32).check().is_ok(),
        "scalar, equal sizes"
    );
    // vmap over the last axis: each batch item is a column of 4 bytes (1 and 2 as uint32).
    let b = Array::from_slice(&[1u8, 2, 0, 0, 0, 0, 0, 0], [4, 2]);
    let got = vmap_flat(
        |xs: &[Array]| Ok(vec![xs[0].view(DType::U32)]),
        &[Some(1)],
        &[0],
    )(&[b])
    .unwrap();
    assert_eq!(got[0].to_vec::<u32>().unwrap(), [1, 2]);
    // No derivative through a bit reinterpretation or fp8 codes.
    let f = Array::from_slice(&[1.0f32, 2.0], [2]);
    assert!(
        grad_flat(
            |xs: &[Array]| Ok(xs[0].view(DType::U32).astype(DType::F32).sum(..)),
            &[0]
        )(std::slice::from_ref(&f))
        .is_err()
    );
    assert!(
        grad_flat(
            |xs: &[Array]| Ok(xs[0].to_fp8().astype(DType::F32).sum(..)),
            &[0]
        )(std::slice::from_ref(&f))
        .is_err()
    );
}
