pub mod array;
pub mod gpu_utils;
pub mod kernels;
pub mod utils;

use gpu_utils::GpuDevice;
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}

#[doc(hidden)]
pub static GPU_DEVICE: OnceLock<Arc<GpuDevice>> = OnceLock::new();
