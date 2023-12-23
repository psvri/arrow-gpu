pub mod array;
pub mod kernels;
pub mod utils;

#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}

use std::sync::{Arc, OnceLock};

use array::GpuDevice;

#[doc(hidden)]
pub static GPU_DEVICE: OnceLock<Arc<GpuDevice>> = OnceLock::new();
