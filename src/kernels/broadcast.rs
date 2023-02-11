use std::sync::Arc;

use async_trait::async_trait;

use crate::array::GpuDevice;

#[async_trait]
pub trait Broadcast<Rhs> {
    type Output;

    async fn broadcast(value: Rhs, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output;
}
