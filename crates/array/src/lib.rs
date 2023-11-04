pub mod array;
pub mod kernels;
pub mod utils;

#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}

use std::sync::Arc;

use array::GpuDevice;
use once_cell::sync::Lazy;
use pollster::FutureExt;

#[doc(hidden)]
pub static GPU_DEVICE: Lazy<Arc<GpuDevice>> = Lazy::new(|| Arc::new(GpuDevice::new().block_on()));
