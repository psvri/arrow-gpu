use std::sync::Arc;

use async_trait::async_trait;

use crate::array::gpu_device::GpuDevice;

#[async_trait]
pub trait Broadcast<Rhs> {
    type Output;

    async fn broadcast(value: Rhs, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output;
}
