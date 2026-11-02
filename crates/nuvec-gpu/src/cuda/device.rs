//! The process-wide CUDA device: ordinal 0, its context, one stream, its facts and the
//! module cache.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cuda_core::sys;

use super::error::{CudaError, Result, check, driver};
use super::memory::Pool;

/// What the GPU is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// Its name (`NVIDIA A40`).
    pub name: String,
    /// Its compute capability (major, minor): (8, 6) for sm_86.
    pub compute_capability: (u32, u32),
    /// Its memory, in bytes.
    pub total_memory: u64,
    /// Its streaming multiprocessors.
    pub multiprocessors: u32,
    /// The driver's CUDA version (13020 for 13.2).
    pub driver_version: u32,
}

impl DeviceInfo {
    /// The architecture name, `sm_86`.
    pub fn arch(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
    }
}

/// A CUDA device with one stream (see the module documentation).
pub struct Device {
    pub(crate) device: Arc<cuda_core::Device>,
    pub(crate) stream: Arc<cuda_core::Stream>,
    info: DeviceInfo,
    pub(crate) pool: Pool,
    /// Loaded PTX modules by source hash, with their functions by name.
    modules: Mutex<HashMap<u64, Arc<cuda_core::Module>>>,
    functions: Mutex<HashMap<(u64, String), Arc<cuda_core::Function>>>,
}

// SAFETY: the context, stream and module handles are thread-safe driver objects (cuda-core
// marks them Send + Sync); the caches are behind mutexes.
unsafe impl Send for Device {}
// SAFETY: as above.
unsafe impl Sync for Device {}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn attribute(device: sys::CUdevice, attr: sys::CUdevice_attribute) -> Result<i32> {
    let mut v = 0;
    // SAFETY: a valid device ordinal from the driver, and an out-parameter.
    check("cuDeviceGetAttribute", unsafe {
        sys::cuDeviceGetAttribute(&mut v, attr, device)
    })?;
    Ok(v)
}

impl Device {
    /// The device at ordinal 0, created on first use; `Err` says why there is none.
    pub fn shared() -> std::result::Result<&'static Device, &'static CudaError> {
        static SHARED: OnceLock<Result<Device>> = OnceLock::new();
        SHARED.get_or_init(Device::create).as_ref()
    }

    fn create() -> Result<Self> {
        if !sys::is_cuda_driver_available() {
            let why = sys::cuda_driver_load_error().map_or_else(
                || "the driver library could not be loaded".into(),
                |e| e.to_string(),
            );
            return Err(CudaError::NoDevice(why));
        }
        let count =
            cuda_core::Device::device_count().map_err(|e| CudaError::NoDevice(e.to_string()))?;
        if count < 1 {
            return Err(CudaError::NoDevice("the driver reports no GPU".into()));
        }
        let device = cuda_core::Device::new(0).map_err(|e| CudaError::NoDevice(e.to_string()))?;
        let cu = device.cu_device();
        let major = attribute(
            cu,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let minor = attribute(
            cu,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?;
        if major < 8 {
            return Err(CudaError::NoDevice(format!(
                "compute capability {major}.{minor}; nuvec needs 8.0 (Ampere) or newer"
            )));
        }
        let managed = attribute(
            cu,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        )?;
        if managed == 0 {
            return Err(CudaError::NoDevice(
                "the GPU has no concurrent managed memory access (Windows, WSL or Tegra)".into(),
            ));
        }
        let mut total = 0usize;
        // SAFETY: a valid device and an out-parameter.
        check("cuDeviceTotalMem", unsafe {
            sys::cuDeviceTotalMem_v2(&mut total, cu)
        })?;
        let mut version = 0;
        // SAFETY: an out-parameter.
        check("cuDriverGetVersion", unsafe {
            sys::cuDriverGetVersion(&mut version)
        })?;
        let info = DeviceInfo {
            name: device.name().map_err(driver("cuDeviceGetName"))?,
            compute_capability: (major as u32, minor as u32),
            total_memory: total as u64,
            multiprocessors: attribute(
                cu,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )? as u32,
            driver_version: version as u32,
        };
        let stream = device.new_stream().map_err(driver("cuStreamCreate"))?;
        Ok(Self {
            pool: Pool::new(info.total_memory as usize, device.cu_ctx()),
            device,
            stream,
            info,
            modules: Mutex::default(),
            functions: Mutex::default(),
        })
    }

    /// What the GPU is.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Waits for every operation submitted to the stream.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the stream is live; waiting has no other precondition.
        unsafe { self.stream.synchronize() }.map_err(driver("cuStreamSynchronize"))
    }

    /// The function `name` of the PTX module `ptx`, loaded (and cached) on first use.
    pub(crate) fn function(&self, ptx: &str, name: &str) -> Result<Arc<cuda_core::Function>> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        ptx.hash(&mut h);
        let key = h.finish();
        if let Some(f) = lock(&self.functions).get(&(key, name.to_string())) {
            return Ok(f.clone());
        }
        // The guard must end before the miss path locks the map again.
        let cached = lock(&self.modules).get(&key).cloned();
        let module = match cached {
            Some(m) => m,
            None => {
                let m =
                    self.device
                        .load_module_from_ptx_src(ptx)
                        .map_err(|e| CudaError::Driver {
                            call: "cuModuleLoadData",
                            message: format!("{e} (PTX of {name})"),
                        })?;
                lock(&self.modules).insert(key, m.clone());
                m
            }
        };
        let f = Arc::new(
            module
                .load_function(name)
                .map_err(driver("cuModuleGetFunction"))?,
        );
        lock(&self.functions).insert((key, name.to_string()), f.clone());
        Ok(f)
    }

    /// Launches `f` on the stream with a 1-D grid and these arguments (each a pointer to
    /// the argument's value).
    pub(crate) fn launch(
        &self,
        f: &cuda_core::Function,
        grid: u32,
        block: u32,
        shared: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        self.device
            .bind_to_thread()
            .map_err(driver("cuCtxSetCurrent"))?;
        // SAFETY: the handle of a function this device's module cache keeps loaded.
        let function = unsafe { f.cu_function() };
        // SAFETY: `function` was loaded in this context; `args` point at values of the
        // kernel's parameter types, alive for the call (the driver copies them); device
        // pointers in them are managed allocations that outlive the stream's use (callers
        // synchronise or keep the buffers alive until they do).
        unsafe {
            cuda_core::launch_kernel(
                function,
                (grid.max(1), 1, 1),
                (block, 1, 1),
                shared,
                self.stream.cu_stream(),
                args,
            )
        }
        .map_err(driver("cuLaunchKernel"))
    }
}
