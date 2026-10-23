//! What a graph node computes.
//!
//! Primitives form a closed enum: equality and hashing — the basis for
//! common-subexpression elimination — are derived, and every per-primitive rule (output
//! inference here; gradient, batching and fusion rules later) is an exhaustive `match`.
//!
//! Primitives require operands of equal shape and dtype. Promotion and broadcasting happen
//! in the ops layer, which inserts `AsType` and `Broadcast` nodes.

use smallvec::{SmallVec, smallvec};

use crate::custom::CustomCall;
use crate::shape::{Shape, Strides};
use crate::view::{self, DimSlice};
use crate::{Array, Category, DType, Error, Result};

/// An element-wise function of one operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// `-x`.
    Negative,
    /// `|x|`; the magnitude for complex numbers.
    Abs,
    /// `e^x`.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Hyperbolic tangent.
    Tanh,
    /// `1 / (1 + e^-x)`.
    Sigmoid,
    /// The complex conjugate (identity for real numbers).
    Conjugate,
    /// The inverse error function.
    ErfInv,
}

impl UnaryOp {
    /// Whether the function computes in floating point (integer inputs are promoted).
    pub(crate) fn is_transcendental(self) -> bool {
        !matches!(self, Self::Negative | Self::Abs | Self::Conjugate)
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Negative => "negative",
            Self::Abs => "abs",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Conjugate => "conjugate",
            Self::ErfInv => "erfinv",
        }
    }
}

/// An element-wise function of two operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Subtract,
    /// `a * b`.
    Multiply,
    /// `a / b` (true division).
    Divide,
    /// The larger operand; NaN propagates.
    Maximum,
    /// The smaller operand; NaN propagates.
    Minimum,
    /// `log(exp(a) + exp(b))`, computed stably (MLX's `logaddexp`); NaN propagates.
    LogAddExp,
    /// `a == b`.
    Equal,
    /// `a != b`.
    NotEqual,
    /// `a < b`.
    Less,
    /// `a <= b`.
    LessEqual,
    /// `a > b`.
    Greater,
    /// `a >= b`.
    GreaterEqual,
}

impl BinaryOp {
    /// Whether the result is boolean.
    pub(crate) fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Equal
                | Self::NotEqual
                | Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
        )
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::LogAddExp => "logaddexp",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
        }
    }
}

/// A reduction over a set of axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceOp {
    /// Sum (identity 0).
    Sum,
    /// Product (identity 1).
    Prod,
    /// Maximum; NaN propagates. No identity.
    Max,
    /// Minimum; NaN propagates. No identity.
    Min,
    /// Logical AND of bools (identity true).
    And,
    /// Logical OR of bools (identity false).
    Or,
}

impl ReduceOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Max => "max",
            Self::Min => "min",
            Self::And => "all",
            Self::Or => "any",
        }
    }
}

/// The position of an extreme value along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArgReduceOp {
    /// Index of the maximum (first occurrence; first NaN wins).
    Max,
    /// Index of the minimum (first occurrence; first NaN wins).
    Min,
}

impl ArgReduceOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Max => "argmax",
            Self::Min => "argmin",
        }
    }
}

/// The combination a cumulative scan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScanOp {
    /// Running sum (`cumsum`).
    Sum,
    /// Running product (`cumprod`).
    Prod,
    /// Running maximum (`cummax`); NaN propagates.
    Max,
    /// Running minimum (`cummin`); NaN propagates.
    Min,
    /// Running `log(Σ exp)` (`logcumsumexp`).
    LogAddExp,
}

impl ScanOp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Sum => "cumsum",
            Self::Prod => "cumprod",
            Self::Max => "cummax",
            Self::Min => "cummin",
            Self::LogAddExp => "logcumsumexp",
        }
    }
}

/// How a scatter combines an update with the value it lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScatterReduce {
    /// The update replaces the value (with repeated indices, one of the updates wins).
    None,
    /// The update is added to the value (repeated indices accumulate).
    Sum,
}

/// An operation recorded in the graph, with its parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Primitive {
    /// Element-wise function of one operand.
    Unary(UnaryOp),
    /// Element-wise function of two operands of equal shape and dtype.
    Binary(BinaryOp),
    /// `where(condition, a, b)`: a boolean condition selects from `a` or `b`.
    Select,
    /// Conversion to another element type.
    AsType(DType),
    /// The same elements in another shape. A view when the layout permits, else a copy.
    Reshape(Vec<usize>),
    /// Axes reordered: output axis `i` is input axis `axes[i]`. A view.
    Transpose(Vec<usize>),
    /// A normalised slice per dimension. A view.
    Slice {
        /// First index per dimension.
        start: Vec<usize>,
        /// Step per dimension (non-zero, may be negative).
        step: Vec<isize>,
        /// Number of selected indices per dimension (the output shape).
        len: Vec<usize>,
    },
    /// Presented as a larger shape with zero strides. A view.
    Broadcast(Vec<usize>),
    /// The input unchanged, but no gradient flows through it. A view.
    StopGradient,
    /// `(base, update)`: a copy of `base` whose region selected as by `Slice` is replaced by
    /// `update` (of shape `len`).
    SliceUpdate {
        /// First index per dimension.
        start: Vec<usize>,
        /// Step per dimension (non-zero, may be negative).
        step: Vec<isize>,
        /// Number of selected indices per dimension (the update's shape).
        len: Vec<usize>,
    },
    /// Reduction over sorted, unique `axes`; reduced dimensions become size 1.
    Reduce {
        /// The reduction.
        op: ReduceOp,
        /// Reduced axes, sorted and unique.
        axes: Vec<usize>,
    },
    /// Index of an extreme value along `axis` (uint32); the axis becomes size 1.
    ArgReduce {
        /// Maximum or minimum.
        op: ArgReduceOp,
        /// The reduced axis.
        axis: usize,
    },
    /// `(src, indices)`: the elements of `src` at `indices` along `axis` (MLX's
    /// `GatherAxis`, `take_along_axis`). `indices` (integers) has `src`'s shape except
    /// along `axis` and gives the output's shape. Negative indices count from the end;
    /// out-of-range indices are clamped.
    GatherAxis {
        /// The indexed axis.
        axis: usize,
    },
    /// `(dst, indices, updates)`: a copy of `dst` with `updates` placed at `indices` along
    /// `axis` (MLX's `ScatterAxis`, `put_along_axis`). `indices` (integers) and `updates`
    /// share a shape, which is `dst`'s except along `axis`. Indices as in `GatherAxis`.
    ScatterAxis {
        /// The indexed axis.
        axis: usize,
        /// How updates combine with `dst`.
        reduce: ScatterReduce,
    },
    /// The input sorted along `axis` (ascending, stable, NaN last).
    Sort {
        /// The sorted axis.
        axis: usize,
    },
    /// The indices (uint32) that sort the input along `axis`, as [`Primitive::Sort`].
    ArgSort {
        /// The sorted axis.
        axis: usize,
    },
    /// A cumulative scan along `axis`.
    Scan {
        /// The combination.
        op: ScanOp,
        /// The scanned axis.
        axis: usize,
        /// Scan from the end of the axis.
        reverse: bool,
        /// Whether each output includes its own element (else it holds the elements
        /// before it, starting from the combination's identity).
        inclusive: bool,
    },
    /// The inputs (one dtype, equal shapes except along `axis`) joined along `axis`.
    Concatenate {
        /// The joined axis.
        axis: usize,
    },
    /// Batched matrix product of `[B..., M, K]` and `[B..., K, N]` with equal batch
    /// dimensions.
    Matmul,
    /// `alpha · (a @ b) + beta · c` for inputs `(a, b, c)`: a `Matmul` whose `c` has the
    /// output's shape. The factors are stored as `f32` bits (so the enum stays `Eq + Hash`).
    AddMM {
        /// `f32::to_bits` of alpha.
        alpha: u32,
        /// `f32::to_bits` of beta.
        beta: u32,
    },
    /// Affine quantization of the last axis in groups (MLX's `quantize`). Outputs the packed
    /// codes (uint32, `bits` per value), the scales and the biases (the input's dtype).
    Quantize {
        /// Values per scale and bias: 32, 64 or 128.
        group_size: usize,
        /// Bits per value: 1–6 or 8.
        bits: u8,
    },
    /// `(w, scales, biases)`: `code · scale + bias` per value, in the scales' dtype.
    Dequantize {
        /// Values per scale and bias.
        group_size: usize,
        /// Bits per value.
        bits: u8,
    },
    /// `(x, w, scales, biases)`: `x · dequantize(w)ᵀ` (`transpose`) or `x · dequantize(w)`.
    /// `w` is two-dimensional, or has `x`'s batch dimensions.
    QuantizedMatmul {
        /// Values per scale and bias.
        group_size: usize,
        /// Bits per value.
        bits: u8,
        /// Whether the dequantized matrix is used transposed (MLX's default).
        transpose: bool,
    },
    /// A fused operation with a dedicated GPU kernel ([`crate::fast`]).
    Fast(crate::fast::imp::FastOp),
    /// Element-wise operations fused into one generated kernel ([`crate::graph::Fused`]), as
    /// `compile` builds them. The inputs share the output's shape.
    Compiled(crate::fused::Fused),
    /// Random bits from threefry2x32 keys of shape `[..., 2]` (uint32): the output has shape
    /// `keys.shape[..-1] + shape` and `width` bytes per element (MLX's layout).
    RandomBits {
        /// The shape drawn per key.
        shape: Vec<usize>,
        /// Bytes per element: 1, 2 or 4.
        width: u8,
    },
    /// A call of a [`crate::custom_function`]: the inputs are the primals followed by the
    /// function's outputs, which this node presents unchanged (views); transforms use the
    /// function's custom rules.
    Custom(CustomCall),
    /// A custom Metal kernel ([`crate::fast::MetalKernel`]).
    MetalKernel(crate::kernel::KernelCall),
    /// A row-contiguous copy of the input into a new buffer. Never simplified away (as a
    /// same-type conversion or identity view would be).
    Copy,
    /// `(x, deps...)`: `x` unchanged (sharing its data), evaluated only after the
    /// `n_deps` other inputs. Gradients flow to `x` only. It orders work, as MLX's
    /// `depends`: a recomputation that depends on cotangents cannot be merged with the
    /// forward pass.
    Depends {
        /// The number of dependencies after `x`.
        n_deps: usize,
    },
    /// Test-only two-output primitive: `(a + b, a * b)`.
    #[cfg(test)]
    AddMultiply,
}

/// The dtype, shape and (for views) strides of one output of a primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputSpec {
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
    /// `None` means row-contiguous.
    pub(crate) strides: Option<Strides>,
}

impl Primitive {
    /// `(alpha, beta)` of an `AddMM`.
    pub(crate) fn addmm_factors(&self) -> (f32, f32) {
        match self {
            Self::AddMM { alpha, beta } => (f32::from_bits(*alpha), f32::from_bits(*beta)),
            _ => (1.0, 0.0),
        }
    }

    /// The operation's name, as used in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unary(op) => op.name(),
            Self::Binary(op) => op.name(),
            Self::Select => "where",
            Self::AsType(_) => "astype",
            Self::Reshape(_) => "reshape",
            Self::Transpose(_) => "transpose",
            Self::Slice { .. } => "slice",
            Self::Broadcast(_) => "broadcast_to",
            Self::StopGradient => "stop_gradient",
            Self::SliceUpdate { .. } => "slice_update",
            Self::Reduce { op, .. } => op.name(),
            Self::ArgReduce { op, .. } => op.name(),
            Self::GatherAxis { .. } => "take_along_axis",
            Self::ScatterAxis {
                reduce: ScatterReduce::None,
                ..
            } => "put_along_axis",
            Self::ScatterAxis { .. } => "scatter_add_along_axis",
            Self::Concatenate { .. } => "concatenate",
            Self::Sort { .. } => "sort",
            Self::ArgSort { .. } => "argsort",
            Self::Scan { op, .. } => op.name(),
            Self::Matmul => "matmul",
            Self::AddMM { .. } => "addmm",
            Self::RandomBits { .. } => "random_bits",
            Self::Fast(op) => op.name(),
            Self::Compiled(_) => "compiled",
            Self::Quantize { .. } => "quantize",
            Self::Dequantize { .. } => "dequantize",
            Self::QuantizedMatmul { .. } => "quantized_matmul",
            Self::Custom(_) => "custom_function",
            Self::MetalKernel(_) => "metal_kernel",
            Self::Copy => "copy",
            Self::Depends { .. } => "depends",
            #[cfg(test)]
            Self::AddMultiply => "add_multiply",
        }
    }

    /// Validates the inputs and infers the outputs, without computing anything.
    pub(crate) fn output_specs(&self, inputs: &[Array]) -> Result<SmallVec<[OutputSpec; 1]>> {
        let computed = |dtype: DType, shape: &[usize]| OutputSpec {
            dtype,
            shape: shape.into(),
            strides: None,
        };
        let spec = match self {
            Self::Unary(op) => {
                let [a] = self.arity::<1>(inputs)?;
                // As the ops layer promotes and rejects (the kernels rely on it).
                let supported = match op {
                    UnaryOp::Negative => a.dtype() != DType::Bool,
                    UnaryOp::Abs | UnaryOp::Conjugate => true,
                    _ => a.dtype().category() == Category::Float,
                };
                if !supported {
                    return Err(Error::unsupported_dtype(op.name(), a.dtype()));
                }
                let dtype = match (op, a.dtype()) {
                    (UnaryOp::Abs, DType::Complex64) => DType::F32,
                    (_, d) => d,
                };
                computed(dtype, a.dims())
            }
            Self::Binary(op) => {
                let [a, b] = self.arity::<2>(inputs)?;
                self.same_shape_and_dtype(a, b)?;
                let category = a.dtype().category();
                let unsupported = match op {
                    BinaryOp::LogAddExp => category != Category::Float,
                    BinaryOp::Divide => !matches!(category, Category::Float | Category::Complex),
                    BinaryOp::Maximum
                    | BinaryOp::Minimum
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual => category == Category::Complex,
                    _ => false,
                };
                if unsupported {
                    return Err(Error::unsupported_dtype(op.name(), a.dtype()));
                }
                computed(
                    if op.is_comparison() {
                        DType::Bool
                    } else {
                        a.dtype()
                    },
                    a.dims(),
                )
            }
            Self::Select => {
                let [c, a, b] = self.arity::<3>(inputs)?;
                self.same_shape_and_dtype(a, b)?;
                if c.dtype() != DType::Bool || c.dims() != a.dims() {
                    return Err(Error::InvalidArgument(
                        "where: condition must be a bool array of the same shape".into(),
                    ));
                }
                computed(a.dtype(), a.dims())
            }
            Self::AsType(dtype) => {
                let [a] = self.arity::<1>(inputs)?;
                computed(*dtype, a.dims())
            }
            Self::Reshape(shape) => {
                let [a] = self.arity::<1>(inputs)?;
                if crate::shape::checked_size(shape)? != a.size() {
                    return Err(Error::InvalidArgument(format!(
                        "reshape: cannot reshape {} elements into shape {shape:?}",
                        a.size()
                    )));
                }
                OutputSpec {
                    dtype: a.dtype(),
                    shape: shape.as_slice().into(),
                    strides: view::nocopy_reshape(a.dims(), a.strides(), shape),
                }
            }
            Self::Transpose(axes) => {
                let [a] = self.arity::<1>(inputs)?;
                let (shape, strides) = view::permute(a.dims(), a.strides(), axes)?;
                OutputSpec {
                    dtype: a.dtype(),
                    shape,
                    strides: Some(strides),
                }
            }
            Self::Slice { len, .. } => {
                let [a] = self.arity::<1>(inputs)?;
                let dims = self.dim_slices(a.ndim())?;
                if !region_fits(&dims, a.dims()) {
                    return Err(Error::InvalidArgument(format!(
                        "slice: region {dims:?} does not fit shape {:?}",
                        a.dims()
                    )));
                }
                let (strides, _) = view::slice_geometry(a.strides(), &dims);
                OutputSpec {
                    dtype: a.dtype(),
                    shape: len.as_slice().into(),
                    strides: Some(strides),
                }
            }
            Self::Broadcast(shape) => {
                let [a] = self.arity::<1>(inputs)?;
                let strides = view::broadcast_strides(a.dims(), a.strides(), shape)?;
                OutputSpec {
                    dtype: a.dtype(),
                    shape: shape.as_slice().into(),
                    strides: Some(strides),
                }
            }
            Self::Copy => {
                let [a] = self.arity::<1>(inputs)?;
                computed(a.dtype(), a.dims())
            }
            Self::Depends { n_deps } => {
                if inputs.len() != n_deps + 1 {
                    return Err(Error::InvalidArgument(format!(
                        "depends: {} dependencies expected, got {}",
                        n_deps,
                        inputs.len().saturating_sub(1)
                    )));
                }
                let a = &inputs[0];
                OutputSpec {
                    dtype: a.dtype(),
                    shape: a.dims().into(),
                    strides: Some(a.strides().into()),
                }
            }
            Self::StopGradient => {
                let [a] = self.arity::<1>(inputs)?;
                OutputSpec {
                    dtype: a.dtype(),
                    shape: a.dims().into(),
                    strides: Some(a.strides().into()),
                }
            }
            Self::SliceUpdate { len, .. } => {
                let [base, update] = self.arity::<2>(inputs)?;
                let dims = self.dim_slices(base.ndim())?;
                if !region_fits(&dims, base.dims()) || update.dims() != len.as_slice() {
                    return Err(Error::InvalidArgument(format!(
                        "slice_update: region {dims:?} does not fit shape {:?} with update {:?}",
                        base.dims(),
                        update.dims()
                    )));
                }
                if base.dtype() != update.dtype() {
                    return Err(Error::dtype_mismatch(
                        "slice_update",
                        base.dtype(),
                        update.dtype(),
                    ));
                }
                computed(base.dtype(), base.dims())
            }
            Self::Reduce { op, axes } => {
                let [a] = self.arity::<1>(inputs)?;
                let shape = self.reduced_shape(a.dims(), axes)?;
                if !axes.windows(2).all(|w| w[0] < w[1]) {
                    return Err(Error::InvalidArgument(format!(
                        "{}: axes {axes:?} must be sorted and unique",
                        self.name()
                    )));
                }
                let supported = match op {
                    ReduceOp::And | ReduceOp::Or => a.dtype() == DType::Bool,
                    ReduceOp::Max | ReduceOp::Min => a.dtype() != DType::Complex64,
                    ReduceOp::Sum | ReduceOp::Prod => true,
                };
                if !supported {
                    return Err(Error::unsupported_dtype(self.name(), a.dtype()));
                }
                let ordered = matches!(op, ReduceOp::Max | ReduceOp::Min);
                if ordered && axes.iter().any(|&d| a.dims()[d] == 0) {
                    return Err(Error::InvalidArgument(format!(
                        "{}: cannot reduce a zero-length axis (no identity)",
                        self.name()
                    )));
                }
                computed(a.dtype(), &shape)
            }
            Self::ArgReduce { axis, .. } => {
                let [a] = self.arity::<1>(inputs)?;
                if a.dtype() == DType::Complex64 {
                    return Err(Error::unsupported_dtype(self.name(), a.dtype()));
                }
                computed(DType::U32, &self.reduced_shape(a.dims(), &[*axis])?)
            }
            Self::GatherAxis { axis } => {
                let [src, indices] = self.arity::<2>(inputs)?;
                self.indexed(src, indices, *axis)?;
                computed(src.dtype(), indices.dims())
            }
            Self::ScatterAxis { axis, .. } => {
                let [dst, indices, updates] = self.arity::<3>(inputs)?;
                self.indexed(dst, indices, *axis)?;
                if updates.dims() != indices.dims() {
                    return Err(Error::shape_mismatch(
                        self.name(),
                        indices.dims(),
                        updates.dims(),
                    ));
                }
                if updates.dtype() != dst.dtype() {
                    return Err(Error::dtype_mismatch(
                        self.name(),
                        dst.dtype(),
                        updates.dtype(),
                    ));
                }
                computed(dst.dtype(), dst.dims())
            }
            Self::Sort { axis } | Self::ArgSort { axis } | Self::Scan { axis, .. } => {
                let [a] = self.arity::<1>(inputs)?;
                if *axis >= a.ndim() {
                    return Err(Error::InvalidArgument(format!(
                        "{}: axis {axis} out of range for {} dimensions",
                        self.name(),
                        a.ndim()
                    )));
                }
                let ordered = matches!(
                    self,
                    Self::Sort { .. }
                        | Self::ArgSort { .. }
                        | Self::Scan {
                            op: ScanOp::Max | ScanOp::Min,
                            ..
                        }
                );
                let float_only = matches!(
                    self,
                    Self::Scan {
                        op: ScanOp::LogAddExp,
                        ..
                    }
                );
                let category = a.dtype().category();
                if (ordered && category == crate::Category::Complex)
                    || (float_only && category != crate::Category::Float)
                {
                    return Err(Error::unsupported_dtype(self.name(), a.dtype()));
                }
                let dtype = match self {
                    Self::ArgSort { .. } => DType::U32,
                    _ => a.dtype(),
                };
                computed(dtype, a.dims())
            }
            Self::Concatenate { axis } => {
                let first = inputs.first().ok_or_else(|| {
                    Error::InvalidArgument("concatenate: no arrays to join".into())
                })?;
                if *axis >= first.ndim() {
                    return Err(Error::InvalidArgument(format!(
                        "concatenate: axis {axis} out of range for {} dimensions",
                        first.ndim()
                    )));
                }
                let mut shape: Shape = first.dims().into();
                for a in &inputs[1..] {
                    let fits = a.ndim() == first.ndim()
                        && a.dims()
                            .iter()
                            .zip(first.dims())
                            .enumerate()
                            .all(|(d, (x, y))| d == *axis || x == y);
                    if !fits {
                        return Err(Error::shape_mismatch("concatenate", first.dims(), a.dims()));
                    }
                    if a.dtype() != first.dtype() {
                        return Err(Error::dtype_mismatch(
                            "concatenate",
                            first.dtype(),
                            a.dtype(),
                        ));
                    }
                    shape[*axis] = shape[*axis].checked_add(a.dims()[*axis]).ok_or_else(|| {
                        Error::InvalidArgument("concatenate: the joined size overflows".into())
                    })?;
                }
                computed(first.dtype(), &shape)
            }
            Self::Matmul | Self::AddMM { .. } => {
                let (a, b) = match self {
                    Self::Matmul => {
                        let [a, b] = self.arity::<2>(inputs)?;
                        (a, b)
                    }
                    _ => {
                        let [a, b, _] = self.arity::<3>(inputs)?;
                        (a, b)
                    }
                };
                let (sa, sb) = (a.dims(), b.dims());
                let n = sa.len();
                let valid =
                    n >= 2 && sb.len() == n && sa[..n - 2] == sb[..n - 2] && sa[n - 1] == sb[n - 2];
                if !valid {
                    return Err(Error::shape_mismatch(self.name(), sa, sb));
                }
                if a.dtype() != b.dtype() {
                    return Err(Error::dtype_mismatch(self.name(), a.dtype(), b.dtype()));
                }
                if !matches!(a.dtype().category(), Category::Float | Category::Complex) {
                    return Err(Error::unsupported_dtype(self.name(), a.dtype()));
                }
                let mut shape: Shape = sa.into();
                shape[n - 1] = sb[n - 1];
                if let Self::AddMM { .. } = self {
                    let c = &inputs[2];
                    if c.dims() != shape.as_slice() || c.dtype() != a.dtype() {
                        return Err(Error::InvalidArgument(format!(
                            "addmm: c must be {} of shape {shape:?}, got {} {:?}",
                            a.dtype(),
                            c.dtype(),
                            c.dims()
                        )));
                    }
                }
                computed(a.dtype(), &shape)
            }
            Self::RandomBits { shape, width } => {
                let [keys] = self.arity::<1>(inputs)?;
                if keys.dtype() != DType::U32 || keys.dims().last() != Some(&2) {
                    return Err(Error::InvalidArgument(format!(
                        "random_bits: keys must be uint32 of shape [..., 2], got {} {:?}",
                        keys.dtype(),
                        keys.dims()
                    )));
                }
                let dtype = match width {
                    1 => DType::U8,
                    2 => DType::U16,
                    4 => DType::U32,
                    _ => {
                        return Err(Error::InvalidArgument(format!(
                            "random_bits: width must be 1, 2 or 4, got {width}"
                        )));
                    }
                };
                let out: Vec<usize> = keys.dims()[..keys.ndim() - 1]
                    .iter()
                    .chain(shape)
                    .copied()
                    .collect();
                computed(dtype, &out)
            }
            Self::Fast(op) => {
                return Ok(op
                    .outputs(inputs)?
                    .into_iter()
                    .map(|(dtype, shape)| computed(dtype, &shape))
                    .collect());
            }
            Self::Compiled(fused) => {
                let dtypes: Vec<DType> = inputs.iter().map(Array::dtype).collect();
                if dtypes != fused.inputs() {
                    return Err(Error::InvalidArgument(format!(
                        "compiled: inputs {dtypes:?}, expected {:?}",
                        fused.inputs()
                    )));
                }
                let shape = inputs[0].dims();
                if let Some(other) = inputs.iter().find(|a| a.dims() != shape) {
                    return Err(Error::shape_mismatch("compiled", shape, other.dims()));
                }
                return Ok(fused
                    .dtypes()
                    .into_iter()
                    .map(|d| computed(d, shape))
                    .collect());
            }
            Self::Quantize { group_size, bits } => {
                let [w] = self.arity::<1>(inputs)?;
                crate::quantized::check_params(self.name(), *group_size, u32::from(*bits))?;
                let n = w.ndim();
                if !matches!(w.dtype(), DType::F16 | DType::BF16 | DType::F32) {
                    return Err(Error::unsupported_dtype("quantize", w.dtype()));
                }
                if n < 2 || w.dims()[n - 1] % group_size != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "quantize: needs at least 2 dimensions with the last divisible by \
                         the group size {group_size}, got shape {:?}",
                        w.dims()
                    )));
                }
                let last = w.dims()[n - 1];
                let with_last = |d: usize| {
                    let mut s: Shape = w.dims().into();
                    s[n - 1] = d;
                    s
                };
                let packed = computed(DType::U32, &with_last(last * usize::from(*bits) / 32));
                let groups = computed(w.dtype(), &with_last(last / group_size));
                return Ok(smallvec![packed, groups.clone(), groups]);
            }
            Self::Dequantize { group_size, bits } => {
                let [w, scales, _] = self.arity::<3>(inputs)?;
                crate::quantized::check_params(self.name(), *group_size, u32::from(*bits))?;
                let mut shape: Shape = w.dims().into();
                let n = shape.len();
                shape[n - 1] = self.quantized_columns(w, &inputs[1..], *group_size, *bits)?;
                computed(scales.dtype(), &shape)
            }
            Self::QuantizedMatmul {
                group_size,
                bits,
                transpose,
            } => {
                let [x, w, scales, _] = self.arity::<4>(inputs)?;
                crate::quantized::check_params(self.name(), *group_size, u32::from(*bits))?;
                let columns = self.quantized_columns(w, &inputs[2..], *group_size, *bits)?;
                let rows = w.dims()[w.ndim() - 2];
                let (inner, outer) = if *transpose {
                    (columns, rows)
                } else {
                    (rows, columns)
                };
                let (xn, wn) = (x.ndim(), w.ndim());
                let batch_ok = wn == 2 || (xn == wn && x.dims()[..xn - 2] == w.dims()[..wn - 2]);
                if xn == 0 || x.dims()[xn - 1] != inner || !batch_ok {
                    return Err(Error::shape_mismatch(
                        "quantized_matmul",
                        x.dims(),
                        w.dims(),
                    ));
                }
                if x.dtype() != scales.dtype() {
                    return Err(Error::dtype_mismatch(
                        "quantized_matmul",
                        x.dtype(),
                        scales.dtype(),
                    ));
                }
                let mut shape: Shape = x.dims().into();
                shape[xn - 1] = outer;
                computed(x.dtype(), &shape)
            }
            Self::MetalKernel(call) => {
                return Ok(call
                    .outputs
                    .iter()
                    .map(|(dtype, shape)| computed(*dtype, shape))
                    .collect());
            }
            Self::Custom(call) => {
                let outputs = inputs.get(call.primals..).unwrap_or_default();
                if outputs.is_empty() {
                    return Err(Error::InvalidArgument(
                        "custom_function: the function must return at least one array".into(),
                    ));
                }
                return Ok(outputs
                    .iter()
                    .map(|a| OutputSpec {
                        dtype: a.dtype(),
                        shape: a.dims().into(),
                        strides: Some(a.strides().into()),
                    })
                    .collect());
            }
            #[cfg(test)]
            Self::AddMultiply => {
                let [a, b] = self.arity::<2>(inputs)?;
                self.same_shape_and_dtype(a, b)?;
                let spec = computed(a.dtype(), a.dims());
                return Ok(smallvec![spec.clone(), spec]);
            }
        };
        Ok(smallvec![spec])
    }

    /// The per-dimension slices of a `Slice` or `SliceUpdate` primitive over `ndim`
    /// dimensions.
    pub(crate) fn dim_slices(&self, ndim: usize) -> Result<Vec<DimSlice>> {
        match self {
            Self::Slice { start, step, len } | Self::SliceUpdate { start, step, len }
                if start.len() == ndim && step.len() == ndim && len.len() == ndim =>
            {
                Ok((0..ndim)
                    .map(|i| DimSlice {
                        start: start[i],
                        step: step[i],
                        len: len[i],
                    })
                    .collect())
            }
            _ => Err(Error::InvalidArgument(
                "slice: one start, step and length per dimension".into(),
            )),
        }
    }

    /// The number of values per row of packed codes `w` (uint32, at least 2-D), after
    /// validating it against `[scales, biases]` (one float dtype, equal shapes, one value per
    /// group), as MLX's `validate_quantized_input`.
    fn quantized_columns(
        &self,
        w: &Array,
        groups: &[Array],
        group_size: usize,
        bits: u8,
    ) -> Result<usize> {
        let (scales, biases) = (&groups[0], &groups[1]);
        let n = w.ndim();
        let invalid =
            |what: String| Err(Error::InvalidArgument(format!("{}: {what}", self.name())));
        if w.dtype() != DType::U32 {
            return invalid(format!("packed weights must be uint32, got {}", w.dtype()));
        }
        if !matches!(scales.dtype(), DType::F16 | DType::BF16 | DType::F32)
            || biases.dtype() != scales.dtype()
            || biases.dims() != scales.dims()
        {
            return invalid(format!(
                "scales and biases must share a float dtype and shape, got {} {:?} and {} {:?}",
                scales.dtype(),
                scales.dims(),
                biases.dtype(),
                biases.dims()
            ));
        }
        let packed = w.dims().last().copied().unwrap_or(0) * 32;
        let groups = scales.dims().last().copied().unwrap_or(0) * group_size;
        if n < 2
            || scales.ndim() != n
            || w.dims()[..n - 1] != scales.dims()[..n - 1]
            || packed != groups * usize::from(bits)
        {
            return invalid(format!(
                "packed weights {:?} and scales {:?} do not match group size {group_size} \
                 and {bits} bits",
                w.dims(),
                scales.dims()
            ));
        }
        Ok(groups)
    }

    /// Validates an indexing operation's array and indices: integer indices with the
    /// array's shape except along `axis`.
    fn indexed(&self, array: &Array, indices: &Array, axis: usize) -> Result<()> {
        if !matches!(
            indices.dtype().category(),
            crate::Category::Signed | crate::Category::Unsigned
        ) {
            return Err(Error::InvalidArgument(format!(
                "{}: indices must be integers, got {}",
                self.name(),
                indices.dtype()
            )));
        }
        let fits = indices.ndim() == array.ndim()
            && axis < array.ndim()
            && indices
                .dims()
                .iter()
                .zip(array.dims())
                .enumerate()
                .all(|(d, (i, a))| d == axis || i == a);
        if !fits {
            return Err(Error::InvalidArgument(format!(
                "{}: indices of shape {:?} do not fit an array of shape {:?} along axis {axis}",
                self.name(),
                indices.dims(),
                array.dims()
            )));
        }
        if array.dims()[axis] == 0 && indices.size() > 0 {
            return Err(Error::InvalidArgument(format!(
                "{}: cannot index an empty axis",
                self.name()
            )));
        }
        Ok(())
    }

    /// `shape` with `axes` set to 1, after validating the axes.
    fn reduced_shape(&self, shape: &[usize], axes: &[usize]) -> Result<Shape> {
        let mut out: Shape = shape.into();
        for &axis in axes {
            *out.get_mut(axis).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{}: axis {axis} out of range for shape {shape:?}",
                    self.name()
                ))
            })? = 1;
        }
        Ok(out)
    }

    fn arity<'a, const N: usize>(&self, inputs: &'a [Array]) -> Result<&'a [Array; N]> {
        inputs.try_into().map_err(|_| {
            Error::InvalidArgument(format!(
                "{} takes {N} inputs, got {}",
                self.name(),
                inputs.len()
            ))
        })
    }

    fn same_shape_and_dtype(&self, a: &Array, b: &Array) -> Result<()> {
        if a.dims() != b.dims() {
            return Err(Error::shape_mismatch(self.name(), a.dims(), b.dims()));
        }
        if a.dtype() != b.dtype() {
            return Err(Error::dtype_mismatch(self.name(), a.dtype(), b.dtype()));
        }
        Ok(())
    }
}

/// A row-contiguous copy of `x` ([`Primitive::Copy`]).
pub(crate) fn copy(x: &Array) -> Result<Array> {
    Ok(Array::from_op(Primitive::Copy, vec![x.clone()])?.swap_remove(0))
}

/// `x`, evaluated after `deps` ([`Primitive::Depends`]).
pub(crate) fn depends(x: &Array, deps: &[Array]) -> Result<Array> {
    let mut inputs = Vec::with_capacity(deps.len() + 1);
    inputs.push(x.clone());
    inputs.extend_from_slice(deps);
    let primitive = Primitive::Depends { n_deps: deps.len() };
    Ok(Array::from_op(primitive, inputs)?.swap_remove(0))
}

/// Whether each dimension's slice selects indices within `shape` with a non-zero step (a
/// zero-length slice starts at most at the end), so that no offset computed from it
/// overflows. The step of a slice of at most one index is never applied (any value).
pub(crate) fn region_fits(dims: &[DimSlice], shape: &[usize]) -> bool {
    dims.len() == shape.len()
        && dims.iter().zip(shape).all(|(d, &n)| {
            if d.step == 0 {
                return false;
            }
            if d.len == 0 {
                return d.start <= n;
            }
            let last = isize::try_from(d.len - 1)
                .ok()
                .and_then(|steps| steps.checked_mul(d.step))
                .and_then(|span| isize::try_from(d.start).ok()?.checked_add(span));
            d.start < n && last.is_some_and(|last| (0..n as isize).contains(&last))
        })
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn hash(p: &Primitive) -> u64 {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    }

    #[test]
    fn equal_primitives_compare_and_hash_equal() {
        let add = Primitive::Binary(BinaryOp::Add);
        assert_eq!(add, Primitive::Binary(BinaryOp::Add));
        assert_eq!(hash(&add), hash(&Primitive::Binary(BinaryOp::Add)));
        assert_ne!(add, Primitive::Binary(BinaryOp::Multiply));
        let s = |start: Vec<usize>| Primitive::Slice {
            start,
            step: vec![1],
            len: vec![2],
        };
        assert_eq!(s(vec![1]), s(vec![1]));
        assert_ne!(s(vec![1]), s(vec![2]));
    }
}
