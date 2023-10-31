pub mod array;
pub mod kernels;
pub mod utils;

#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}

#[cfg(test)]
mod test {
    use once_cell::sync::Lazy;
    use pollster::FutureExt;
    use std::sync::Arc;

    use crate::array::GpuDevice;

    pub static GPU_DEVICE: Lazy<Arc<GpuDevice>> =
        Lazy::new(|| Arc::new(GpuDevice::new().block_on()));
}
