//! The Metal runtime on `objc2-metal` (macOS only).

mod allocator;
mod buffer;
mod command;
pub mod profile;
mod residency;

use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

use allocator::Pool;
pub use buffer::Buffer;
pub use command::{Arg, CommandBuffer, Pending, Submitted, Timeline};
use nuvec_kernels::MetallibMode;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};

use crate::info::parse_architecture;
use crate::{DeviceInfo, GpuError, MemoryStats, Result};

type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// A function constant value for [`Device::pipeline_with_constants`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constant {
    /// A `bool` constant.
    Bool(bool),
    /// An `int` constant.
    Int(i32),
}

/// A Metal GPU with its command queue, kernel library and pipeline cache. Cheap to clone.
#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

struct Inner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    info: DeviceInfo,
    library: OnceLock<Result<Library>>,
    pipelines: Mutex<HashMap<String, Pipeline>>,
    pool: Arc<Pool>,
    in_flight: InFlight,
}

/// The number of committed command buffers that have not completed yet.
#[derive(Default)]
pub(crate) struct InFlight {
    count: Mutex<usize>,
    done: Condvar,
}

impl InFlight {
    pub(crate) fn begin(&self) {
        *self.count.lock().unwrap_or_else(PoisonError::into_inner) += 1;
    }

    pub(crate) fn end(&self) {
        let mut count = self.count.lock().unwrap_or_else(PoisonError::into_inner);
        *count = count.saturating_sub(1);
        self.done.notify_all();
    }

    fn get(&self) -> usize {
        *self.count.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait_below(&self, max: usize) {
        let mut count = self.count.lock().unwrap_or_else(PoisonError::into_inner);
        while *count >= max.max(1) {
            count = self
                .done
                .wait(count)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Device {
    /// The system's default Metal device, or `None` if there is none (for example in some
    /// virtual machines).
    pub fn system_default() -> Option<Device> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        let architecture = device.architecture().name().to_string();
        let (generation, class) = parse_architecture(&architecture);
        let info = DeviceInfo {
            name: device.name().to_string(),
            architecture,
            generation,
            class,
            recommended_working_set: device.recommendedMaxWorkingSetSize(),
            max_threads_per_threadgroup: device.maxThreadsPerThreadgroup().width,
        };
        let ram = objc2_foundation::NSProcessInfo::processInfo().physicalMemory();
        let options = objc2_metal::MTLResourceOptions::StorageModeShared;
        let placeholder = device.newBufferWithLength_options(16, options);
        let residency = residency::Residency::new(residency::attach(&device, &queue));
        let pool = Arc::new(Pool::new(
            info.recommended_working_set,
            ram,
            placeholder,
            residency,
        ));
        Some(Device {
            inner: Arc::new(Inner {
                device,
                queue,
                info,
                library: OnceLock::new(),
                pipelines: Mutex::new(HashMap::new()),
                pool,
                in_flight: InFlight::default(),
            }),
        })
    }

    /// The process-wide default device, created on first use: one allocator, kernel cache
    /// and queue shared by everything that uses the GPU.
    pub fn shared() -> Option<&'static Device> {
        static SHARED: OnceLock<Option<Device>> = OnceLock::new();
        SHARED.get_or_init(Device::system_default).as_ref()
    }

    /// Memory use and limits of this device's allocator.
    pub fn memory(&self) -> MemoryStats {
        self.inner.pool.stats()
    }

    /// Sets the memory limit (allocations beyond it fail) and returns the previous one.
    pub fn set_memory_limit(&self, bytes: usize) -> usize {
        self.inner.pool.set_memory_limit(bytes)
    }

    /// Sets the cache limit, trimming the cache to it, and returns the previous one.
    pub fn set_cache_limit(&self, bytes: usize) -> usize {
        self.inner.pool.set_cache_limit(bytes)
    }

    /// Sets the wired limit and returns the previous one (MLX's `set_wired_limit`; 0 by
    /// default). Buffers, mapped files included, are kept resident up to this many bytes
    /// while GPU work runs, instead of being made resident per command buffer, which can
    /// page memory in again under pressure. Lowering it unwires buffers until the wired set
    /// fits. A limit above the recommended working set is an error. Without residency sets
    /// (before macOS 15, or on GPUs before Metal 3) the limit is recorded but nothing is
    /// wired.
    pub fn set_wired_limit(&self, bytes: usize) -> Result<usize> {
        if bytes as u64 > self.inner.info.recommended_working_set {
            return Err(GpuError::InvalidArgument(format!(
                "wired limit {bytes} exceeds the recommended working set ({} bytes)",
                self.inner.info.recommended_working_set
            )));
        }
        Ok(self.inner.pool.set_wired_limit(bytes))
    }

    /// Whether buffers can be wired on this system (residency sets: macOS 15 and a Metal 3
    /// GPU).
    pub fn can_wire(&self) -> bool {
        self.inner.pool.can_wire()
    }

    /// Releases every cached buffer.
    pub fn clear_cache(&self) {
        self.inner.pool.clear_cache();
    }

    /// Command buffers committed on this device that have not completed yet.
    pub fn in_flight(&self) -> usize {
        self.inner.in_flight.get()
    }

    /// Blocks until fewer than `max` command buffers are in flight (back-pressure for
    /// callers that submit without waiting).
    pub fn wait_for_capacity(&self, max: usize) {
        self.inner.in_flight.wait_below(max);
    }

    pub(crate) fn flights(&self) -> &InFlight {
        &self.inner.in_flight
    }

    /// Resets the peak to the current active memory.
    pub fn reset_peak(&self) {
        self.inner.pool.reset_peak();
    }

    pub(crate) fn pool(&self) -> &Arc<Pool> {
        &self.inner.pool
    }

    /// What this GPU is and can do.
    pub fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }

    /// The compute pipeline for the kernel `name` from nuvec's kernel library, compiled on
    /// first use and cached.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline> {
        self.pipeline_with(name, &[])
    }

    /// The pipeline for kernel `name` specialised with boolean function constants
    /// `(index, value)`, compiled on first use and cached per name and constants.
    pub fn pipeline_with(&self, name: &str, constants: &[(usize, bool)]) -> Result<Pipeline> {
        let constants: Vec<(usize, Constant)> = constants
            .iter()
            .map(|&(i, v)| (i, Constant::Bool(v)))
            .collect();
        self.pipeline_with_constants(name, &constants)
    }

    /// The pipeline for kernel `name` specialised with boolean or integer function constants
    /// `(index, value)`, compiled on first use and cached per name and constants.
    pub fn pipeline_with_constants(
        &self,
        name: &str,
        constants: &[(usize, Constant)],
    ) -> Result<Pipeline> {
        let key = constants.iter().fold(name.to_owned(), |mut k, (i, v)| {
            match v {
                Constant::Bool(b) => k.push_str(&format!("#{i}={}", u8::from(*b))),
                Constant::Int(n) => k.push_str(&format!("#{i}=i{n}")),
            }
            k
        });
        let mut cache = self
            .inner
            .pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let library = self.library()?;
        let ns_name = NSString::from_str(name);
        let function = if constants.is_empty() {
            library
                .newFunctionWithName(&ns_name)
                .ok_or_else(|| GpuError::MissingKernel(name.to_owned()))?
        } else {
            let values = MTLFunctionConstantValues::new();
            for &(index, value) in constants {
                match value {
                    // SAFETY: `b` is a live local, which Metal copies during the call; `bool`
                    // matches `MTLDataType::Bool` (one byte).
                    Constant::Bool(b) => unsafe {
                        values.setConstantValue_type_atIndex(
                            NonNull::from(&b).cast(),
                            MTLDataType::Bool,
                            index,
                        );
                    },
                    // SAFETY: as above; `i32` matches `MTLDataType::Int`.
                    Constant::Int(n) => unsafe {
                        values.setConstantValue_type_atIndex(
                            NonNull::from(&n).cast(),
                            MTLDataType::Int,
                            index,
                        );
                    },
                }
            }
            library
                .newFunctionWithName_constantValues_error(&ns_name, &values)
                .map_err(|e| GpuError::Pipeline {
                    name: key.clone(),
                    message: e.localizedDescription().to_string(),
                })?
        };
        let state = self
            .inner
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| GpuError::Pipeline {
                name: key.clone(),
                message: e.localizedDescription().to_string(),
            })?;
        let pipeline = Pipeline {
            state,
            name: key.as_str().into(),
        };
        cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// The pipeline for kernel `name` from Metal source that `source` generates, compiled
    /// at runtime on first use and cached by name (so a name must always mean the same
    /// source). The source may include nuvec's headers (`#include "common.h"`); it is
    /// compiled with IEEE-conformant math, as nuvec's own kernels are.
    pub fn pipeline_from_source(
        &self,
        name: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Pipeline> {
        let mut cache = self
            .inner
            .pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }
        let failed = |message: String| GpuError::Pipeline {
            name: name.to_owned(),
            message,
        };
        let source = inline_sources(&[("generated.metal", &source())]);
        let options = objc2_metal::MTLCompileOptions::new();
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
        let library = self
            .inner
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| failed(e.localizedDescription().to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| GpuError::MissingKernel(name.to_owned()))?;
        let state = self
            .inner
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| failed(e.localizedDescription().to_string()))?;
        let pipeline = Pipeline {
            state,
            name: name.into(),
        };
        cache.insert(name.to_owned(), pipeline.clone());
        Ok(pipeline)
    }

    /// nuvec's kernel library, loaded once.
    fn library(&self) -> Result<&Library> {
        self.inner
            .library
            .get_or_init(|| load_library(&self.inner.device))
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.inner.queue
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("info", &self.inner.info)
            .finish_non_exhaustive()
    }
}

/// Loads the precompiled library, or compiles the embedded sources (with their headers
/// inlined, as the runtime compiler has no include path for them).
fn load_library(device: &ProtocolObject<dyn MTLDevice>) -> Result<Library> {
    let error = |e: Retained<objc2_foundation::NSError>| {
        GpuError::LibraryLoad(e.localizedDescription().to_string())
    };
    match nuvec_kernels::MODE {
        MetallibMode::Precompiled => {
            let bytes = nuvec_kernels::metallib().ok_or(GpuError::NoKernels)?;
            device
                .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(bytes))
                .map_err(error)
        }
        MetallibMode::RuntimeSource => {
            let source = inline_sources(nuvec_kernels::sources());
            // IEEE-conformant math, matching the precompiled library's `-fno-fast-math`.
            let options = objc2_metal::MTLCompileOptions::new();
            options.setMathMode(objc2_metal::MTLMathMode::Safe);
            device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
                .map_err(error)
        }
        _ => Err(GpuError::NoKernels),
    }
}

/// All `.metal` sources as one translation unit, each `#include "header.h"` of one of
/// nuvec's headers replaced by the header's text on first use.
fn inline_sources(sources: &[(&str, &str)]) -> String {
    let headers: HashMap<&str, &str> = nuvec_kernels::sources()
        .iter()
        .chain(sources)
        .copied()
        .chain([("common.h", nuvec_kernels::COMMON_H)])
        .filter(|(p, _)| p.ends_with(".h"))
        .collect();
    let mut included = HashSet::new();
    let mut out = String::new();
    for (_, source) in sources.iter().filter(|(p, _)| p.ends_with(".metal")) {
        for line in source.lines() {
            let header = line
                .trim()
                .strip_prefix("#include \"")
                .and_then(|r| r.strip_suffix('"'));
            match header.and_then(|h| headers.get(h).map(|text| (h, text))) {
                Some((h, text)) => {
                    if included.insert(h) {
                        out.push_str(text);
                        out.push('\n');
                    }
                }
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// A compiled compute kernel. Cheap to clone.
#[derive(Clone)]
pub struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    name: Arc<str>,
}

impl Pipeline {
    /// The kernel's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The most threads one threadgroup of this kernel may have.
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.state.maxTotalThreadsPerThreadgroup()
    }

    /// The SIMD-group width (threads that execute in lockstep).
    pub fn execution_width(&self) -> usize {
        self.state.threadExecutionWidth()
    }

    /// Whether two handles refer to the same compiled pipeline.
    pub fn same_as(&self, other: &Pipeline) -> bool {
        Retained::as_ptr(&self.state) == Retained::as_ptr(&other.state)
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::inline_sources;

    #[test]
    fn inlines_each_header_once() {
        let sources = [
            ("a.metal", "#include \"h.h\"\nkernel A;"),
            ("b.metal", "  #include \"h.h\"\nkernel B;"),
            ("h.h", "#pragma once\ninline int f();"),
        ];
        let out = inline_sources(&sources);
        assert_eq!(out.matches("inline int f();").count(), 1);
        assert!(out.contains("kernel A;") && out.contains("kernel B;"));
        assert!(!out.contains("#include \"h.h\""));
    }
}
