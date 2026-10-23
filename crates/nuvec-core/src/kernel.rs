//! Custom Metal kernels from source, as MLX's `mx.fast.metal_kernel`.
//!
//! A kernel is a name, the names of its inputs and outputs, and the body of a Metal kernel
//! function. Calling it with arrays generates the function's signature (as MLX does):
//!
//! - an input of at least one dimension is `const device T* name`, a 0-d input
//!   `const constant T& name`; `name_shape`, `name_strides` and `name_ndim` are added when
//!   the body uses them;
//! - an output is `device T* name` (`device atomic<T>* name` with atomic outputs);
//! - thread attributes the body uses (`thread_position_in_grid`, `thread_index_in_simdgroup`,
//!   ...) become parameters;
//! - template arguments become `constexpr` values and `typedef`s at the top of the body.
//!
//! The kernel runs on the GPU only, with one thread per grid position (`dispatchThreads`).
//! It has no derivative or batching rule; wrap it in [`crate::custom_function`] (or [`crate::custom_function_flat`]) to give it
//! one.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::primitive::Primitive;
use crate::{Array, DType, DeviceKind, Error, Result, default_stream};

/// A template argument of a custom kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemplateArg {
    /// `constexpr int name = value;`
    Int(i64),
    /// `constexpr bool name = value;`
    Bool(bool),
    /// `typedef type name;`
    DType(DType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelSpec {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    source: String,
    header: String,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
}

/// A Metal kernel built from source, as MLX's `mx.fast.metal_kernel`: a name, the names
/// of its inputs and outputs, and the body of a kernel function. Calling it generates the
/// function's signature as MLX does:
///
/// - an input of at least one dimension is `const device T* name`, a 0-d input
///   `const constant T& name`; `name_shape`, `name_strides` and `name_ndim` are added when
///   the body uses them;
/// - an output is `device T* name` (`device atomic<T>* name` with atomic outputs);
/// - thread attributes the body uses (`thread_position_in_grid`,
///   `thread_index_in_simdgroup`, ...) become parameters;
/// - template arguments become `constexpr` values and `typedef`s at the top of the body;
///   MLX's type names (`bfloat16_t`, `float16_t`) and `elem_to_loc` are defined.
///
/// It runs on the GPU only, one thread per grid position (`dispatchThreads`), and has no
/// derivative or batching rule; wrap it in [`crate::custom_function`] (or [`crate::custom_function_flat`]) to give it one.
///
/// The source is trusted code, run as written: nuvec checks the inputs, outputs and
/// launch configuration, but not what the body does with them. A body that reads or
/// writes outside its inputs' and outputs' elements accesses other GPU memory, which is
/// undefined behaviour (as with any `unsafe` code); pass only source you would compile
/// into your program.
///
/// ```
/// use nuvec_core::fast::{KernelConfig, MetalKernel};
/// use nuvec_core::{Array, DType, Device, Result, array, default_stream};
///
/// # fn main() -> Result<()> {
/// # if default_stream().device != Device::gpu(0) || (Array::from(1.0f32) * 2.0).eval().is_err() { return Ok(()); }
/// let exp = MetalKernel::new(
///     "myexp",
///     &["inp"],
///     &["out"],
///     "uint elem = thread_position_in_grid.x;\nout[elem] = metal::exp(inp[elem]);",
/// );
/// let x = array![0.0, 1.0];
/// let config = KernelConfig { grid: [2, 1, 1], threadgroup: [2, 1, 1], ..Default::default() };
/// let y = exp.call(&[x], &[(&[2], DType::F32)], &config).remove(0);
/// assert!((y.to_vec::<f32>()?[1] - std::f32::consts::E).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MetalKernel {
    spec: KernelSpec,
}

/// How a [`MetalKernel`] is launched.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KernelConfig {
    /// Threads in the grid, `[x, y, z]`.
    pub grid: [usize; 3],
    /// Threads per threadgroup, `[x, y, z]` (at most 1,024 in all).
    pub threadgroup: [usize; 3],
    /// Template arguments, by name.
    pub template: Vec<(String, TemplateArg)>,
    /// A value every output holds before the kernel runs (else they are uninitialised).
    pub init_value: Option<f64>,
}

/// A recorded call of a custom kernel.
#[derive(Debug, Clone)]
pub struct KernelCall {
    pub(crate) spec: Arc<KernelSpec>,
    pub(crate) template: Vec<(String, TemplateArg)>,
    pub(crate) grid: [usize; 3],
    pub(crate) threadgroup: [usize; 3],
    pub(crate) outputs: Vec<(DType, Vec<usize>)>,
    /// The outputs' initial values are the last inputs.
    pub(crate) initialised: bool,
}

impl PartialEq for KernelCall {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec
            && self.template == other.template
            && self.grid == other.grid
            && self.threadgroup == other.threadgroup
            && self.outputs == other.outputs
            && self.initialised == other.initialised
    }
}

impl Eq for KernelCall {}

impl Hash for KernelCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.spec.hash(state);
        self.template.hash(state);
        self.grid.hash(state);
        self.threadgroup.hash(state);
        self.outputs.hash(state);
        self.initialised.hash(state);
    }
}

impl MetalKernel {
    /// A kernel named `name` (letters, digits and underscores) whose body `source` reads
    /// `inputs` and writes `outputs` (in this order).
    ///
    /// Each distinct instantiation (source, template arguments, input and output types and
    /// layouts) compiles one pipeline, cached for the life of the process: call a kernel
    /// with a bounded set of template arguments (not, say, a sequence length) to keep the
    /// cache bounded.
    pub fn new(name: &str, inputs: &[&str], outputs: &[&str], source: &str) -> Self {
        Self {
            spec: KernelSpec {
                name: name.to_owned(),
                inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
                outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
                source: source.to_owned(),
                header: String::new(),
                ensure_row_contiguous: true,
                atomic_outputs: false,
            },
        }
    }

    /// Code placed before the kernel function (helper functions, includes).
    pub fn header(mut self, header: &str) -> Self {
        self.spec.header = header.to_owned();
        self
    }

    /// Whether inputs are copied to row-major order first (the default, as MLX); otherwise
    /// the body indexes them through `name_strides`.
    pub fn ensure_row_contiguous(mut self, yes: bool) -> Self {
        self.spec.ensure_row_contiguous = yes;
        self
    }

    /// Whether outputs are `device atomic<T>*` (to be updated with `atomic_*` functions).
    pub fn atomic_outputs(mut self, yes: bool) -> Self {
        self.spec.atomic_outputs = yes;
        self
    }

    /// Runs the kernel on `inputs`, producing one array per output of the given shape and
    /// dtype. Invalid arguments (names, counts, dtypes, sizes) give failed arrays.
    #[track_caller]
    pub fn call(
        &self,
        inputs: &[Array],
        outputs: &[(&[i32], DType)],
        config: &KernelConfig,
    ) -> Vec<Array> {
        let ins: Vec<&Array> = inputs.iter().collect();
        Array::sticky_vec(
            "metal_kernel",
            std::panic::Location::caller(),
            &ins,
            outputs.len().max(1),
            || {
                let shapes = outputs
                    .iter()
                    .map(|(shape, _)| {
                        let spec: Vec<i64> = shape.iter().map(|&d| i64::from(d)).collect();
                        crate::axes::dims(&spec)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let outs: Vec<(&[usize], DType)> = shapes
                    .iter()
                    .zip(outputs)
                    .map(|(s, (_, d))| (s.as_slice(), *d))
                    .collect();
                self.call_impl(inputs, &outs, config)
            },
        )
    }

    /// Runs the kernel on `inputs`, producing arrays of the given shapes and dtypes (`usize`
    /// shapes; the implementation behind [`MetalKernel::call`]).
    pub(crate) fn call_impl(
        &self,
        inputs: &[Array],
        outputs: &[(&[usize], DType)],
        config: &KernelConfig,
    ) -> Result<Vec<Array>> {
        let invalid = |msg: String| Err(Error::InvalidArgument(format!("metal_kernel: {msg}")));
        let spec = &self.spec;
        let word = |s: &String| {
            !s.is_empty()
                && !s.starts_with(|c: char| c.is_ascii_digit())
                && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        if !word(&spec.name) || !spec.inputs.iter().chain(&spec.outputs).all(word) {
            return invalid("names must be identifiers".into());
        }
        if inputs.len() != spec.inputs.len() || outputs.len() != spec.outputs.len() {
            return invalid(format!(
                "{} inputs and {} outputs expected, got {} and {}",
                spec.inputs.len(),
                spec.outputs.len(),
                inputs.len(),
                outputs.len()
            ));
        }
        if outputs.is_empty() {
            return invalid("a kernel needs at least one output".into());
        }
        if config.grid.contains(&0) || config.threadgroup.contains(&0) {
            return invalid("grid and threadgroup sizes must be positive".into());
        }
        if config.threadgroup.iter().product::<usize>() > 1024 {
            return invalid("at most 1,024 threads per threadgroup".into());
        }
        let all = inputs
            .iter()
            .map(Array::dtype)
            .chain(outputs.iter().map(|o| o.1));
        if let Some(dtype) = all
            .chain(config.template.iter().filter_map(|(_, a)| match a {
                TemplateArg::DType(d) => Some(*d),
                _ => None,
            }))
            .find(|&d| msl_type(d).is_none())
        {
            return Err(Error::unsupported_dtype("metal_kernel", dtype));
        }
        if default_stream().device.kind != DeviceKind::Gpu {
            return Err(Error::UnsupportedDevice(default_stream().device));
        }
        let mut operands = inputs.to_vec();
        if let Some(v) = config.init_value {
            for &(shape, dtype) in outputs {
                operands.push(crate::ops::imp::broadcast_to(
                    &Array::from(v).try_astype(dtype)?,
                    shape,
                )?);
            }
        }
        let call = KernelCall {
            spec: Arc::new(KernelSpec {
                name: spec.name.clone(),
                inputs: spec.inputs.clone(),
                outputs: spec.outputs.clone(),
                source: spec.source.clone(),
                header: spec.header.clone(),
                ensure_row_contiguous: spec.ensure_row_contiguous,
                atomic_outputs: spec.atomic_outputs,
            }),
            template: config.template.clone(),
            grid: config.grid,
            threadgroup: config.threadgroup,
            outputs: outputs.iter().map(|&(s, d)| (d, s.to_vec())).collect(),
            initialised: config.init_value.is_some(),
        };
        Array::from_op(Primitive::MetalKernel(call), operands)
    }
}

/// MLX's name of a dtype in kernel source.
pub(crate) fn msl_type(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::Bool => "bool",
        DType::U8 => "uint8_t",
        DType::U16 => "uint16_t",
        DType::U32 => "uint32_t",
        DType::U64 => "uint64_t",
        DType::I8 => "int8_t",
        DType::I16 => "int16_t",
        DType::I32 => "int32_t",
        DType::I64 => "int64_t",
        DType::F16 => "float16_t",
        DType::BF16 => "bfloat16_t",
        DType::F32 => "float",
        DType::F64 | DType::Complex64 => return None,
    })
}

/// Thread attributes a body may use, with their types.
const ATTRIBUTES: [(&str, &str); 13] = [
    ("thread_position_in_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threads_per_threadgroup", "uint3"),
    ("thread_index_in_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("threads_per_simdgroup", "uint"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("thread_index_in_quadgroup", "uint"),
];

/// Whether `word` occurs in `text` as an identifier.
fn uses(text: &str, word: &str) -> bool {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.match_indices(word).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + word.len()..].chars().next();
        !before.is_some_and(ident) && !after.is_some_and(ident)
    })
}

/// What a kernel's signature needs about each input.
pub(crate) struct InputLayout {
    pub(crate) dtype: DType,
    pub(crate) scalar: bool,
}

/// Buffer roles in a generated signature, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Binding {
    /// Input `i`'s data.
    Input(usize),
    /// Input `i`'s shape (`int` per dimension).
    Shape(usize),
    /// Input `i`'s strides (`int64_t` per dimension).
    Strides(usize),
    /// Input `i`'s number of dimensions (`int`).
    Ndim(usize),
    /// Output `i`.
    Output(usize),
}

impl KernelCall {
    /// The kernel's own inputs (the rest of a call's operands are initial outputs).
    pub(crate) fn input_count(&self) -> usize {
        self.spec.inputs.len()
    }

    /// Whether inputs are bound row-major.
    pub(crate) fn row_contiguous_inputs(&self) -> bool {
        self.spec.ensure_row_contiguous
    }

    /// The buffers of the generated signature, in order.
    pub(crate) fn bindings(&self, inputs: &[InputLayout]) -> Vec<Binding> {
        let spec = &self.spec;
        let mut out = Vec::new();
        for (i, (name, layout)) in spec.inputs.iter().zip(inputs).enumerate() {
            out.push(Binding::Input(i));
            if layout.scalar {
                continue;
            }
            if uses(&spec.source, &format!("{name}_shape")) {
                out.push(Binding::Shape(i));
            }
            if uses(&spec.source, &format!("{name}_strides")) {
                out.push(Binding::Strides(i));
            }
            if uses(&spec.source, &format!("{name}_ndim")) {
                out.push(Binding::Ndim(i));
            }
        }
        out.extend((0..spec.outputs.len()).map(Binding::Output));
        out
    }

    /// The pipeline name and the Metal source of the kernel for inputs laid out as given.
    pub(crate) fn source(&self, inputs: &[InputLayout]) -> (String, String) {
        let spec = &self.spec;
        let ty = |d: DType| msl_type(d).expect("checked when called");
        let mut params = Vec::new();
        for (k, binding) in self.bindings(inputs).into_iter().enumerate() {
            params.push(match binding {
                Binding::Input(i) => {
                    let (name, layout) = (&spec.inputs[i], &inputs[i]);
                    match layout.scalar {
                        true => format!(
                            "const constant {}& {name} [[buffer({k})]]",
                            ty(layout.dtype)
                        ),
                        false => {
                            format!("const device {}* {name} [[buffer({k})]]", ty(layout.dtype))
                        }
                    }
                }
                Binding::Shape(i) => {
                    format!(
                        "const constant int* {}_shape [[buffer({k})]]",
                        spec.inputs[i]
                    )
                }
                Binding::Strides(i) => {
                    format!(
                        "const constant int64_t* {}_strides [[buffer({k})]]",
                        spec.inputs[i]
                    )
                }
                Binding::Ndim(i) => {
                    format!(
                        "const constant int& {}_ndim [[buffer({k})]]",
                        spec.inputs[i]
                    )
                }
                Binding::Output(i) => {
                    let t = ty(self.outputs[i].0);
                    match spec.atomic_outputs {
                        true => format!("device atomic<{t}>* {} [[buffer({k})]]", spec.outputs[i]),
                        false => format!("device {t}* {} [[buffer({k})]]", spec.outputs[i]),
                    }
                }
            });
        }
        for (attr, t) in ATTRIBUTES {
            if uses(&spec.source, attr) {
                params.push(format!("{t} {attr} [[{attr}]]"));
            }
        }
        let template: String = self
            .template
            .iter()
            .map(|(name, arg)| match arg {
                TemplateArg::Int(v) => format!("  constexpr int {name} = {v};\n"),
                TemplateArg::Bool(v) => format!("  constexpr bool {name} = {v};\n"),
                TemplateArg::DType(d) => format!("  typedef {} {name};\n", ty(*d)),
            })
            .collect();
        let body = format!(
            "{PRELUDE}{}\n[[kernel]] void KERNEL_NAME(\n  {}) {{\n{template}{}\n}}\n",
            spec.header,
            params.join(",\n  "),
            spec.source
        );
        let mut h = DefaultHasher::new();
        body.hash(&mut h);
        let name = format!("nuvec_custom_{}_{:016x}", spec.name, h.finish());
        (name.clone(), body.replace("KERNEL_NAME", &name))
    }
}

/// Definitions every custom kernel sees: Metal's standard library and MLX's type names and
/// `elem_to_loc`.
const PRELUDE: &str = r#"#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_atomic>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;

// The element of a strided array at row-major position `elem` (MLX's `elem_to_loc`).
inline int64_t elem_to_loc(uint elem, constant const int* shape, constant const int64_t* strides, int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += int64_t(elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}
"#;
