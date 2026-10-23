//! Custom Metal kernels (`fast::MetalKernel`): signature generation (inputs, shapes and
//! strides, 0-d inputs, outputs, atomics, attributes, templates), initial values, errors,
//! and MLX source compatibility (mlx-lm's gated delta kernel, unchanged).

#![cfg(target_os = "macos")]

mod common;

use common::Rng;
use nuvec_core::fast::{self, KernelConfig, MetalKernel, TemplateArg};
use nuvec_core::ops;
use nuvec_core::*;

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    !cfg!(miri) && (Array::from(1.0f32) * 2.0).eval().is_ok()
}

/// Whether a call failed: every output is a failed array.
fn failed(outs: &[Array]) -> bool {
    !outs.is_empty() && outs.iter().all(Array::is_failed)
}

fn launch(n: usize) -> KernelConfig {
    KernelConfig {
        grid: [n, 1, 1],
        threadgroup: [n.min(256), 1, 1],
        ..Default::default()
    }
}

#[test]
fn templates_scalars_and_multiple_outputs() {
    if !gpu() {
        return;
    }
    let k = MetalKernel::new(
        "scaled",
        &["inp", "offset"],
        &["out", "twice"],
        "uint i = thread_position_in_grid.x;\n\
         out[i] = static_cast<T>(inp[i] * N) + static_cast<T>(offset);\n\
         twice[i] = 2 * inp[i];",
    );
    let x = array![1.0, 2.0, 3.0];
    let offset = Array::from(0.5f32);
    let mut config = launch(3);
    config.template = vec![
        ("T".into(), TemplateArg::DType(DType::F16)),
        ("N".into(), TemplateArg::Int(3)),
    ];
    let outs = k.call(
        &[x, offset],
        &[(&[3], DType::F16), (&[3], DType::F32)],
        &config,
    );
    assert_eq!(outs[0].dtype(), DType::F16);
    let v = outs[0].astype(DType::F32).to_vec::<f32>().unwrap();
    assert_eq!(v, [3.5, 6.5, 9.5]);
    assert_eq!(outs[1].to_vec::<f32>().unwrap(), [2.0, 4.0, 6.0]);
}

#[test]
fn strided_inputs_are_read_through_their_strides() {
    if !gpu() {
        return;
    }
    let k = MetalKernel::new(
        "strided",
        &["inp"],
        &["out"],
        "uint i = thread_position_in_grid.x;\n\
         out[i] = inp[elem_to_loc(i, inp_shape, inp_strides, inp_ndim)];",
    )
    .ensure_row_contiguous(false);
    let a = Array::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), [2, 3]);
    let t = a.transpose([1, 0]);
    let out = k.call(
        std::slice::from_ref(&t),
        &[(&[3, 2], DType::F32)],
        &launch(6),
    );
    assert_eq!(out[0].to_vec::<f32>().unwrap(), t.to_vec::<f32>().unwrap());
}

#[test]
fn atomic_outputs_start_from_their_initial_value() {
    if !gpu() {
        return;
    }
    let k = MetalKernel::new(
        "total",
        &["inp"],
        &["out"],
        "atomic_fetch_add_explicit(&out[0], inp[thread_position_in_grid.x], memory_order_relaxed);",
    )
    .atomic_outputs(true);
    let x = ops::full([1000], 0.5f32);
    let mut config = launch(1000);
    config.init_value = Some(1.0);
    let out = k.call(&[x], &[(&[1], DType::F32)], &config);
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [501.0]);
}

#[test]
fn mlx_lm_gated_delta_kernel_runs_unchanged() {
    if !gpu() {
        return;
    }
    // mlx-lm's `gated_delta.py` generic kernel source (scalar gates, no mask).
    let source = r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    "#;
    let kernel = MetalKernel::new(
        "gated_delta_step",
        &["q", "k", "v", "g", "beta", "state_in", "T"],
        &["y", "state_out"],
        source,
    );
    let mut rng = Rng::new(0x81);
    let (b, t, hk, hv, dk, dv) = (1usize, 5usize, 2usize, 4usize, 64usize, 16usize);
    let r = |rng: &mut Rng, s: &[usize]| {
        let n: usize = s.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 / 8.0).collect();
        Array::from_slice(&v, s)
    };
    let unit = |a: Array| {
        let n = (&a * &a).sum_keepdims(-1).sqrt();
        &a / &n
    };
    let q = r(&mut rng, &[b, t, hk, dk]);
    let k = unit(r(&mut rng, &[b, t, hk, dk]));
    let v = r(&mut rng, &[b, t, hv, dv]);
    let g = r(&mut rng, &[b, t, hv]) + 0.8;
    let beta = r(&mut rng, &[b, t, hv]) + 0.5;
    let state = r(&mut rng, &[b, hv, dv, dk]);
    let config = KernelConfig {
        grid: [32, dv, b * hv],
        threadgroup: [32, 4, 1],
        template: vec![
            ("InT".into(), TemplateArg::DType(DType::F32)),
            ("StT".into(), TemplateArg::DType(DType::F32)),
            ("Dk".into(), TemplateArg::Int(dk as i64)),
            ("Dv".into(), TemplateArg::Int(dv as i64)),
            ("Hk".into(), TemplateArg::Int(hk as i64)),
            ("Hv".into(), TemplateArg::Int(hv as i64)),
        ],
        init_value: None,
    };
    let inputs = [
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
        Array::from(t as i32),
    ];
    let (b, t, hv, dv, dk) = (b as i32, t as i32, hv as i32, dv as i32, dk as i32);
    let outs = kernel.call(
        &inputs,
        &[
            (&[b, t, hv, dv], DType::F32),
            (&[b, hv, dv, dk], DType::F32),
        ],
        &config,
    );
    let (y, s) = fast::gated_delta(&q, &k, &v, &g, &beta, &state);
    for (got, want) in outs.iter().zip([y, s]) {
        let (a, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
        for (i, (x, z)) in a.iter().zip(&w).enumerate() {
            assert!((x - z).abs() <= 1e-5 * z.abs().max(1.0), "{i}: {x} vs {z}");
        }
    }
}

#[test]
fn invalid_calls_are_rejected() {
    if !gpu() {
        return;
    }
    let k = MetalKernel::new("copy1", &["inp"], &["out"], "out[0] = inp[0];");
    let x = array![1.0];
    let one = launch(1);
    assert!(
        failed(&k.call(&[], &[(&[1], DType::F32)], &one)),
        "input count"
    );
    assert!(
        failed(&k.call(std::slice::from_ref(&x), &[], &one)),
        "output count"
    );
    assert!(
        failed(&k.call(std::slice::from_ref(&x), &[(&[1], DType::F64)], &one)),
        "float64"
    );
    let big = KernelConfig {
        grid: [2048, 1, 1],
        threadgroup: [2048, 1, 1],
        ..Default::default()
    };
    assert!(failed(&k.call(
        std::slice::from_ref(&x),
        &[(&[1], DType::F32)],
        &big
    )));
    assert!(failed(
        &MetalKernel::new("bad name", &["inp"], &["out"], "").call(
            std::slice::from_ref(&x),
            &[(&[1], DType::F32)],
            &one
        )
    ));
    // No derivative: an error, not a silent zero.
    let f = |xs: &[Array]| Ok(k.call(xs, &[(&[1], DType::F32)], &one));
    assert!(vjp_flat(f, std::slice::from_ref(&x), &[array![1.0]]).is_err());
    // GPU only.
    set_default_stream(Stream::default_for(Device::cpu()));
    let outs = k.call(std::slice::from_ref(&x), &[(&[1], DType::F32)], &one);
    assert!(matches!(
        outs[0].clone().check(),
        Err(Error::UnsupportedDevice(_))
    ));
    set_default_stream(Stream::default_for(Device::gpu(0)));
}
