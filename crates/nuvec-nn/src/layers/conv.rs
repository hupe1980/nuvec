//! Convolution and pooling layers (MLX's `nn.Conv1d` … `nn.Conv3d`, `nn.ConvTranspose1d` …
//! `nn.ConvTranspose3d`, `nn.MaxPool1d` … `nn.AvgPool3d`), channels last: inputs are
//! `[N, spatial…, C]`.

use nuvec::ops::{self, ConvArg, PadMode};
use nuvec::{Array, OpErrorKind, Param, Step};

use super::uniform_init;
use crate::Module;

/// The weight `[out, kernel…, in / groups]`, drawn uniformly from `±√(1 / fan)`.
#[track_caller]
fn kernel_init(out_channels: i32, kernel: &[i32], in_per_group: i32, fan: i32) -> Param {
    let mut shape = vec![out_channels];
    shape.extend(kernel);
    shape.push(in_per_group);
    Param::new(uniform_init(fan, &shape))
}

macro_rules! conv_layer {
    ($(#[$doc:meta])* $name:ident, $n:literal, $mlx:literal, |$x:ident, $l:ident| $call:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Module)]
        pub struct $name {
            /// The kernels, `[out_channels, kernel…, in_channels / groups]`.
            pub weight: Param,
            /// The bias, `[out_channels]` (zeros), if any.
            pub bias: Option<Param>,
            /// Positions between outputs, per spatial axis.
            #[tree(skip)]
            pub stride: [i32; $n],
            /// Zeros before and after the input, per spatial axis.
            #[tree(skip)]
            pub padding: [i32; $n],
            /// Positions between kernel taps, per spatial axis.
            #[tree(skip)]
            pub dilation: [i32; $n],
            /// Channel groups.
            #[tree(skip)]
            pub groups: i32,
        }

        impl $name {
            #[doc = concat!("A layer with a bias, initialised as MLX's `", $mlx, "`: the weight drawn")]
            /// uniformly from `±√(1 / (in_channels · ∏ kernel))`, the bias zeros. Stride,
            /// padding and dilation default to 1, 0 and 1.
            #[track_caller]
            pub fn new(in_channels: i32, out_channels: i32, kernel_size: impl Into<ConvArg<$n>>) -> Self {
                Self::grouped(in_channels, out_channels, kernel_size, 1)
            }

            /// [`Self::new`] with `groups` channel groups: the weight is `[out_channels,
            /// kernel…, in_channels / groups]` and its scale counts `in_channels / groups`.
            #[track_caller]
            pub fn grouped(
                in_channels: i32,
                out_channels: i32,
                kernel_size: impl Into<ConvArg<$n>>,
                groups: i32,
            ) -> Self {
                let kernel = kernel_size.into().0;
                let per_group = in_channels / groups.max(1);
                let fan = per_group * kernel.iter().product::<i32>();
                Self {
                    weight: kernel_init(out_channels, &kernel, per_group, fan),
                    bias: Some(Param::new(ops::zeros([out_channels], nuvec::DType::F32))),
                    stride: [1; $n],
                    padding: [0; $n],
                    dilation: [1; $n],
                    groups,
                }
            }

            /// Sets the stride.
            pub fn stride(mut self, stride: impl Into<ConvArg<$n>>) -> Self {
                self.stride = stride.into().0;
                self
            }

            /// Sets the zero padding.
            pub fn padding(mut self, padding: impl Into<ConvArg<$n>>) -> Self {
                self.padding = padding.into().0;
                self
            }

            /// Sets the kernel dilation.
            pub fn dilation(mut self, dilation: impl Into<ConvArg<$n>>) -> Self {
                self.dilation = dilation.into().0;
                self
            }

            /// The layer without its bias (no draw changes: the bias is zeros).
            pub fn without_bias(mut self) -> Self {
                self.bias = None;
                self
            }

            /// The convolution of `x` (`[N, spatial…, in_channels]`) plus the bias.
            #[track_caller]
            pub fn forward(&self, x: &Array) -> Array {
                let ($x, $l) = (x, self);
                let y = $call;
                match &self.bias {
                    Some(b) => y.add(b),
                    None => y,
                }
            }
        }
    };
}

conv_layer!(
    /// A 1-D convolution over `[N, L, C]` (MLX's `nn.Conv1d`).
    Conv1d, 1, "nn.Conv1d",
    |x, l| ops::conv1d(x, &l.weight, l.stride[0], l.padding[0], l.dilation[0], l.groups)
);
conv_layer!(
    /// A 2-D convolution over `[N, H, W, C]` (MLX's `nn.Conv2d`).
    Conv2d, 2, "nn.Conv2d",
    |x, l| ops::conv2d(x, &l.weight, l.stride, l.padding, l.dilation, l.groups)
);
conv_layer!(
    /// A 3-D convolution over `[N, D, H, W, C]` (MLX's `nn.Conv3d`; grouped beyond MLX).
    Conv3d, 3, "nn.Conv3d",
    |x, l| ops::conv3d(x, &l.weight, l.stride, l.padding, l.dilation, l.groups)
);

macro_rules! conv_transpose_layer {
    ($(#[$doc:meta])* $name:ident, $n:literal, $mlx:literal, |$x:ident, $l:ident| $call:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Module)]
        pub struct $name {
            /// The kernels, `[out_channels, kernel…, in_channels / groups]`.
            pub weight: Param,
            /// The bias, `[out_channels]` (zeros), if any.
            pub bias: Option<Param>,
            /// Positions between the input's elements in the output, per spatial axis.
            #[tree(skip)]
            pub stride: [i32; $n],
            /// Positions removed from both ends of the output, per spatial axis.
            #[tree(skip)]
            pub padding: [i32; $n],
            /// Positions between kernel taps, per spatial axis.
            #[tree(skip)]
            pub dilation: [i32; $n],
            /// Positions added to one end of the output, per spatial axis.
            #[tree(skip)]
            pub output_padding: [i32; $n],
            /// Channel groups.
            #[tree(skip)]
            pub groups: i32,
        }

        impl $name {
            #[doc = concat!("A layer with a bias, initialised as MLX's `", $mlx, "`: the weight drawn")]
            /// uniformly from `±√(1 / (in_channels · ∏ kernel))`, the bias zeros. Stride,
            /// padding, dilation and output padding default to 1, 0, 1 and 0.
            #[track_caller]
            pub fn new(in_channels: i32, out_channels: i32, kernel_size: impl Into<ConvArg<$n>>) -> Self {
                Self::grouped(in_channels, out_channels, kernel_size, 1)
            }

            /// [`Self::new`] with `groups` channel groups (beyond MLX's layer): the weight is
            /// `[out_channels, kernel…, in_channels / groups]`.
            #[track_caller]
            pub fn grouped(
                in_channels: i32,
                out_channels: i32,
                kernel_size: impl Into<ConvArg<$n>>,
                groups: i32,
            ) -> Self {
                let kernel = kernel_size.into().0;
                let per_group = in_channels / groups.max(1);
                let fan = per_group * kernel.iter().product::<i32>();
                Self {
                    weight: kernel_init(out_channels, &kernel, per_group, fan),
                    bias: Some(Param::new(ops::zeros([out_channels], nuvec::DType::F32))),
                    stride: [1; $n],
                    padding: [0; $n],
                    dilation: [1; $n],
                    output_padding: [0; $n],
                    groups,
                }
            }

            /// Sets the stride.
            pub fn stride(mut self, stride: impl Into<ConvArg<$n>>) -> Self {
                self.stride = stride.into().0;
                self
            }

            /// Sets the padding.
            pub fn padding(mut self, padding: impl Into<ConvArg<$n>>) -> Self {
                self.padding = padding.into().0;
                self
            }

            /// Sets the kernel dilation.
            pub fn dilation(mut self, dilation: impl Into<ConvArg<$n>>) -> Self {
                self.dilation = dilation.into().0;
                self
            }

            /// Sets the output padding.
            pub fn output_padding(mut self, output_padding: impl Into<ConvArg<$n>>) -> Self {
                self.output_padding = output_padding.into().0;
                self
            }

            /// The layer without its bias (no draw changes: the bias is zeros).
            pub fn without_bias(mut self) -> Self {
                self.bias = None;
                self
            }

            /// The transposed convolution of `x` (`[N, spatial…, in_channels]`) plus the bias.
            #[track_caller]
            pub fn forward(&self, x: &Array) -> Array {
                let ($x, $l) = (x, self);
                let y = $call;
                match &self.bias {
                    Some(b) => y.add(b),
                    None => y,
                }
            }
        }
    };
}

conv_transpose_layer!(
    /// A 1-D transposed convolution over `[N, L, C]` (MLX's `nn.ConvTranspose1d`).
    ConvTranspose1d, 1, "nn.ConvTranspose1d",
    |x, l| ops::conv_transpose1d(
        x, &l.weight, l.stride[0], l.padding[0], l.dilation[0], l.output_padding[0], l.groups,
    )
);
conv_transpose_layer!(
    /// A 2-D transposed convolution over `[N, H, W, C]` (MLX's `nn.ConvTranspose2d`).
    ConvTranspose2d, 2, "nn.ConvTranspose2d",
    |x, l| ops::conv_transpose2d(x, &l.weight, l.stride, l.padding, l.dilation, l.output_padding, l.groups)
);
conv_transpose_layer!(
    /// A 3-D transposed convolution over `[N, D, H, W, C]` (MLX's `nn.ConvTranspose3d`).
    ConvTranspose3d, 3, "nn.ConvTranspose3d",
    |x, l| ops::conv_transpose3d(x, &l.weight, l.stride, l.padding, l.dilation, l.output_padding, l.groups)
);

/// The taps of `x` (`[N, spatial…, C]`) at kernel offset `k`: every `stride`-th position
/// from `k`, `out` of them per axis.
#[track_caller]
fn tap(x: &Array, k: &[i32], stride: &[i32], out: &[i32]) -> Array {
    let r = |i: usize| (k[i]..k[i] + (out[i] - 1) * stride[i] + 1).step(stride[i]);
    match k.len() {
        1 => x.i((.., r(0), ..)),
        2 => x.i((.., r(0), r(1), ..)),
        _ => x.i((.., r(0), r(1), r(2), ..)),
    }
}

/// MLX's `_Pool`: `x` padded with `value`, its windows (a reshape where they tile the
/// input exactly, else the stacked taps) reduced with `max` or `mean`.
#[track_caller]
fn pool(
    name: &'static str,
    x: &Array,
    kernel: &[i32],
    stride: &[i32],
    padding: &[i32],
    max: bool,
) -> Array {
    let d = kernel.len();
    if kernel.iter().chain(stride).any(|&v| v <= 0) || padding.iter().any(|&p| p < 0) {
        return Array::failure(
            name,
            OpErrorKind::InvalidArgument(format!(
                "kernel_size and stride must be positive and padding non-negative, got {kernel:?}, \
                 {stride:?} and {padding:?}"
            )),
        );
    }
    if x.ndim() != d + 2 {
        return Array::failure(
            name,
            OpErrorKind::InvalidArgument(format!(
                "the input must be [N, {d} spatial, C], got shape {:?}",
                x.shape()
            )),
        );
    }
    let x = if padding.iter().any(|&p| p > 0) {
        let mut widths = vec![(0, 0)];
        widths.extend(padding.iter().map(|&p| (p, p)));
        widths.push((0, 0));
        let value = if max { f64::NEG_INFINITY } else { 0.0 };
        ops::pad(x, &widths, PadMode::Constant, value)
    } else {
        x.clone()
    };
    let shape = x.shape();
    let size = |i: usize| shape[1 + i];
    let reduce = |a: &Array, axes: &[i32]| if max { a.max(axes) } else { a.mean(axes) };
    if (0..d).all(|i| stride[i] == kernel[i] && size(i) % kernel[i] == 0) {
        // [N, o0, k0, o1, k1, …, C], reduced over the k axes.
        let mut split = vec![shape[0]];
        for (i, &k) in kernel.iter().enumerate() {
            split.extend([size(i) / k, k]);
        }
        split.push(shape[d + 1]);
        let axes: Vec<i32> = (0..d).map(|i| 2 + 2 * i as i32).collect();
        return reduce(&x.reshape(&split), &axes);
    }
    let out: Vec<i32> = (0..d)
        .map(|i| {
            if size(i) < kernel[i] {
                0
            } else {
                (size(i) - kernel[i]) / stride[i] + 1
            }
        })
        .collect();
    let taps: i32 = kernel.iter().product();
    let offsets = (0..taps).map(|t| {
        let mut rem = t;
        let mut k = vec![0; d];
        for i in (0..d).rev() {
            k[i] = rem % kernel[i];
            rem /= kernel[i];
        }
        k
    });
    let windows: Vec<Array> = offsets.map(|k| tap(&x, &k, stride, &out)).collect();
    reduce(&ops::stack(&windows, -2), &[-2])
}

macro_rules! pool_layer {
    ($(#[$doc:meta])* $name:ident, $n:literal, $max:literal, $op:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Module)]
        pub struct $name {
            /// The window, per spatial axis.
            #[tree(skip)]
            pub kernel_size: [i32; $n],
            /// Positions between windows, per spatial axis (the window by default).
            #[tree(skip)]
            pub stride: [i32; $n],
            /// Padding before and after each spatial axis.
            #[tree(skip)]
            pub padding: [i32; $n],
        }

        impl $name {
            /// Pooling over `kernel_size` windows, the stride the window, no padding.
            pub fn new(kernel_size: impl Into<ConvArg<$n>>) -> Self {
                let kernel_size = kernel_size.into().0;
                Self {
                    kernel_size,
                    stride: kernel_size,
                    padding: [0; $n],
                }
            }

            /// Sets the stride.
            pub fn stride(mut self, stride: impl Into<ConvArg<$n>>) -> Self {
                self.stride = stride.into().0;
                self
            }

            /// Sets the padding.
            pub fn padding(mut self, padding: impl Into<ConvArg<$n>>) -> Self {
                self.padding = padding.into().0;
                self
            }

            /// The pooled `x` (`[N, spatial…, C]`).
            #[track_caller]
            pub fn forward(&self, x: &Array) -> Array {
                pool($op, x, &self.kernel_size, &self.stride, &self.padding, $max)
            }
        }
    };
}

pool_layer!(
    /// The maximum over 1-D windows of `[N, L, C]`, padded with `-∞` (MLX's `nn.MaxPool1d`).
    MaxPool1d, 1, true, "max_pool1d"
);
pool_layer!(
    /// The maximum over 2-D windows of `[N, H, W, C]`, padded with `-∞` (MLX's
    /// `nn.MaxPool2d`).
    MaxPool2d, 2, true, "max_pool2d"
);
pool_layer!(
    /// The maximum over 3-D windows of `[N, D, H, W, C]`, padded with `-∞` (MLX's
    /// `nn.MaxPool3d`).
    MaxPool3d, 3, true, "max_pool3d"
);
pool_layer!(
    /// The mean over 1-D windows of `[N, L, C]`, padded with zeros that count (MLX's
    /// `nn.AvgPool1d`).
    AvgPool1d, 1, false, "avg_pool1d"
);
pool_layer!(
    /// The mean over 2-D windows of `[N, H, W, C]`, padded with zeros that count (MLX's
    /// `nn.AvgPool2d`).
    AvgPool2d, 2, false, "avg_pool2d"
);
pool_layer!(
    /// The mean over 3-D windows of `[N, D, H, W, C]`, padded with zeros that count (MLX's
    /// `nn.AvgPool3d`).
    AvgPool3d, 3, false, "avg_pool3d"
);
