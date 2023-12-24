use std::sync::Arc;

use crate::array::gpu_device::GpuDevice;

pub trait Broadcast<Rhs> {
    type Output;

    fn broadcast(value: Rhs, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output;
}
