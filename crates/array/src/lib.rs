pub mod array;
pub mod gpu_utils;
pub mod kernels;
pub mod utils;

use gpu_utils::GpuDevice;
use std::sync::{Arc, LazyLock};

/// Enum of errors
#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}

#[doc(hidden)]
pub static GPU_DEVICE: LazyLock<Arc<GpuDevice>> = LazyLock::new(|| Arc::new(GpuDevice::new()));
