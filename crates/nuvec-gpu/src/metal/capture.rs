//! GPU captures (MLX's `mx.metal.start_capture`): the command buffers committed between
//! start and stop, written to a `.gputrace` document for Xcode.

use std::path::Path;

use objc2::rc::autoreleasepool;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use super::Device;
use crate::{GpuError, Result};

impl Device {
    /// Starts capturing this device's GPU work into `path`, which must end in `.gputrace`
    /// and not exist. Metal allows captures only with `MTL_CAPTURE_ENABLED=1` in the
    /// environment (or under Xcode); one capture runs at a time.
    pub fn start_capture(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.extension().is_none_or(|e| e != "gputrace") {
            return Err(GpuError::Capture(format!(
                "the capture file must end in .gputrace: {}",
                path.display()
            )));
        }
        if path.exists() {
            return Err(GpuError::Capture(format!(
                "the capture file already exists: {}",
                path.display()
            )));
        }
        autoreleasepool(|_| {
            // SAFETY: the shared manager is a process-wide singleton, valid for the call.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
                return Err(GpuError::Capture(
                    "captures to a file are disabled: set MTL_CAPTURE_ENABLED=1".into(),
                ));
            }
            let descriptor = MTLCaptureDescriptor::new();
            let device: &objc2::runtime::AnyObject = self.inner.device.as_ref();
            // SAFETY: the capture object is a live MTLDevice, the kind Metal expects.
            unsafe { descriptor.setCaptureObject(Some(device)) };
            descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
            let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
            descriptor.setOutputURL(Some(&url));
            manager
                .startCaptureWithDescriptor_error(&descriptor)
                .map_err(|e| GpuError::Capture(e.localizedDescription().to_string()))
        })
    }

    /// Stops the running capture, writing its document (nothing if none runs).
    pub fn stop_capture(&self) {
        autoreleasepool(|_| {
            // SAFETY: as in `start_capture`.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if manager.isCapturing() {
                manager.stopCapture();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_paths_and_the_environment_are_checked() {
        let Some(d) = Device::shared() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("nuvec-capture-{}", std::process::id()));
        let e = d.start_capture(dir.join("trace.txt")).unwrap_err();
        assert!(e.to_string().contains(".gputrace"), "{e}");
        let trace = dir.join("trace.gputrace");
        if std::env::var_os("MTL_CAPTURE_ENABLED").is_none() {
            let e = d.start_capture(&trace).unwrap_err();
            assert!(e.to_string().contains("MTL_CAPTURE_ENABLED"), "{e}");
        } else {
            std::fs::create_dir_all(&dir).unwrap();
            d.start_capture(&trace).unwrap();
            let again = d.start_capture(dir.join("again.gputrace")).unwrap_err();
            assert!(matches!(again, GpuError::Capture(_)), "{again}");
            d.stop_capture();
            assert!(trace.exists());
            let e = d.start_capture(&trace).unwrap_err();
            assert!(e.to_string().contains("already exists"), "{e}");
            std::fs::remove_dir_all(&dir).unwrap();
        }
        d.stop_capture();
    }
}
