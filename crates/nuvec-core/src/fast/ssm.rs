//! Mamba-2's one-token state update (mlx-lm's `ssm_update` kernel).

use std::sync::LazyLock;

use crate::kernel::{KernelConfig, MetalKernel, TemplateArg};
use crate::{Array, DType, DeviceKind, Error, NewAxis, Result, default_stream, ops};

/// mlx-lm 0.32.2's `ssm_kernel` (`mlx_lm/models/ssm.py`, MIT, Copyright © 2025 Apple Inc.),
/// unchanged, so the GPU gives its bits.
const SOURCE: &str = "
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
";

static KERNEL: LazyLock<MetalKernel> = LazyLock::new(|| {
    MetalKernel::new(
        "ssm_kernel",
        &["X", "A_log", "B", "C", "D", "dt", "state_in"],
        &["out", "state_out"],
        SOURCE,
    )
});

pub(crate) fn ssm_update(
    x: &Array,
    a_log: &Array,
    b: &Array,
    c: &Array,
    d: &Array,
    dt: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let invalid = |what: String| Error::InvalidArgument(format!("ssm_update: {what}"));
    let [n, one, h, dh] = <[usize; 4]>::try_from(x.dims()).map_err(|_| {
        invalid(format!(
            "x must be [batch, 1, heads, head_dim], got {:?}",
            x.dims()
        ))
    })?;
    let [bn, bone, g, ds] = <[usize; 4]>::try_from(b.dims()).map_err(|_| {
        invalid(format!(
            "b must be [batch, 1, groups, state], got {:?}",
            b.dims()
        ))
    })?;
    if one != 1 || bone != 1 || bn != n || c.dims() != b.dims() {
        return Err(invalid(format!(
            "one token: x {:?}, b {:?}, c {:?}",
            x.dims(),
            b.dims(),
            c.dims()
        )));
    }
    if g == 0 || h % g != 0 || ds % 32 != 0 {
        return Err(invalid(format!(
            "heads ({h}) must be a multiple of groups ({g}), state ({ds}) of 32"
        )));
    }
    if state.dims() != [n, h, dh, ds] || a_log.dims() != [h] || d.dims() != [h] {
        return Err(invalid(format!(
            "state {:?} (want [{n}, {h}, {dh}, {ds}]), a_log {:?}, d {:?} (want [{h}])",
            state.dims(),
            a_log.dims(),
            d.dims()
        )));
    }
    if dt.size() != n * h {
        return Err(invalid(format!(
            "dt has {} elements, want {}",
            dt.size(),
            n * h
        )));
    }
    let (t, u) = (x.dtype(), state.dtype());
    if default_stream().device.kind == DeviceKind::Gpu {
        let config = KernelConfig {
            grid: [32, dh, h * n],
            threadgroup: [32, 8, 1],
            template: vec![
                ("T".into(), TemplateArg::DType(t)),
                ("U".into(), TemplateArg::DType(u)),
                ("Dh".into(), TemplateArg::Int(dh as i64)),
                ("Ds".into(), TemplateArg::Int(ds as i64)),
                ("H".into(), TemplateArg::Int(h as i64)),
                ("G".into(), TemplateArg::Int((h / g) as i64)),
            ],
            ..KernelConfig::default()
        };
        let (ni, hi, dhi, dsi) = (n as i32, h as i32, dh as i32, ds as i32);
        let outs = KERNEL.call(
            &[
                x.clone(),
                a_log.clone(),
                b.clone(),
                c.clone(),
                d.clone(),
                dt.clone(),
                state.clone(),
            ],
            &[(&[ni, 1, hi, dhi], t), (&[ni, hi, dhi, dsi], u)],
            &config,
        );
        let [y, s] = <[Array; 2]>::try_from(outs).map_err(|_| invalid("two outputs".into()))?;
        return Ok((y, s));
    }
    // The CPU: the kernel's arithmetic in float32 (its sum over the state in another order).
    let f = DType::F32;
    let (n, h, dh, ds, g) = (n as i32, h as i32, dh as i32, ds as i32, g as i32);
    let per = h / g;
    let x32 = x.astype(f).reshape([n, h, dh, 1]);
    let dt32 = dt.astype(f).reshape([n, h, 1, 1]);
    let a = a_log.astype(f).exp().negative().reshape([1, h, 1, 1]);
    let da = a.multiply(&dt32).exp();
    let heads = |m: &Array| {
        ops::repeat(&m.reshape([n, g, 1, ds]), per, 1)
            .astype(f)
            .reshape([n, h, 1, ds])
    };
    let next = da
        .multiply(state.astype(f))
        .add(x32.multiply(&dt32).multiply(heads(b)));
    let y = next
        .multiply(heads(c))
        .sum(-1)
        .add(
            x32.reshape([n, h, dh])
                .multiply(d.astype(f).reshape([1, h, 1])),
        )
        .reshape([n, 1, h, dh])
        .astype(t);
    Ok((y, next.astype(u)))
}

/// mlx-lm's `segsum` without a mask: `out[.., i, j] = Σ_{j < k ≤ i} x[.., k]` below the
/// diagonal, and its cumulative sums elsewhere, as mlx-lm computes it.
fn segsum(x: &Array) -> Array {
    let l = x.dims().last().copied().unwrap_or_default() as i32;
    let x = ops::repeat(&x.expand_dims(-1), l, -1);
    ops::tril(&x, -1).cumsum(-2)
}

/// mlx-lm's chunk length.
const STEP: usize = 256;

pub(crate) fn ssm_scan(
    x: &Array,
    a_log: &Array,
    b: &Array,
    c: &Array,
    d: &Array,
    dt: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    let invalid = |what: String| Error::InvalidArgument(format!("ssm_scan: {what}"));
    let [bn, l, h, dh] = <[usize; 4]>::try_from(x.dims()).map_err(|_| {
        invalid(format!(
            "x must be [batch, tokens, heads, head_dim], got {:?}",
            x.dims()
        ))
    })?;
    let [_, _, g, ds] = <[usize; 4]>::try_from(b.dims()).map_err(|_| {
        invalid(format!(
            "b must be [batch, tokens, groups, state], got {:?}",
            b.dims()
        ))
    })?;
    if b.dims() != [bn, l, g, ds] || c.dims() != b.dims() || dt.dims() != [bn, l, h] {
        return Err(invalid(format!(
            "x {:?}, b {:?}, c {:?}, dt {:?}",
            x.dims(),
            b.dims(),
            c.dims(),
            dt.dims()
        )));
    }
    if g == 0 || h % g != 0 {
        return Err(invalid(format!(
            "heads ({h}) must be a multiple of groups ({g})"
        )));
    }
    let (bi, hi, dhi, gi, dsi) = (bn as i32, h as i32, dh as i32, g as i32, ds as i32);
    let repeats = hi / gi;
    // As mlx-lm's `ssm_attn` (after `compute_dt`), in its order.
    let a = a_log.exp().negative().astype(dt.dtype());
    let dta = dt.multiply(a.reshape([1, 1, -1]));
    let dtx = dt.reshape([bi, l as i32, hi, 1]).multiply(x);
    let mut state = state.cloned();
    let mut ys = Vec::new();
    for i in (0..l).step_by(STEP) {
        let s = STEP.min(l - i) as i32;
        let (i, end) = (i as i32, i as i32 + s);
        let dtx = dtx.i((.., i..end));
        let dta = dta.i((.., i..end));
        let bc = b.i((.., i..end)).transpose([0, 2, 3, 1]);
        let cc = c.i((.., i..end));
        let cb = ops::repeat(&cc.swapaxes(1, 2).matmul(&bc), repeats, 1);
        let decay = segsum(&dta.swapaxes(1, 2)).exp();
        let attention = ops::tril(&cb.multiply(&decay), 0);
        let mut y = attention.matmul(&dtx.swapaxes(1, 2)).swapaxes(1, 2);
        let last = decay.i((.., .., -1.., ..)).transpose([0, 3, 1, 2]);
        let bh = ops::repeat(&bc, repeats, 1).swapaxes(2, 3);
        let mut next = dtx
            .multiply(&last)
            .swapaxes(1, 2)
            .swapaxes(2, 3)
            .matmul(&bh);
        if let Some(state) = &state {
            let cum = dta.cumsum(-2).exp();
            next = next.add(cum.i((.., -1, .., NewAxis, NewAxis)).multiply(state));
            let cr = cc.reshape([bi, s, gi, 1, dsi, 1]);
            let prev = state
                .reshape([bi, 1, gi, repeats, dhi, dsi])
                .matmul(&cr)
                .squeeze(-1);
            let prev = ops::flatten(&prev, 2, 3);
            y = y.add(cum.expand_dims(-1).multiply(&prev));
        }
        ys.push(y.astype(x.dtype()));
        state = Some(next);
    }
    let y = ops::concatenate(&ys, 1).add(x.multiply(d.reshape([1, 1, hi, 1])));
    Ok((y, state.expect("at least one token")))
}
