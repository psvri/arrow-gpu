use std::sync::Arc;

use crate::gpu_utils::{ArrowComputePipeline, GpuDevice};

/// Trait for broadcasting a single value for the whole array
pub trait Broadcast<Rhs>: Sized {
    /// Broadcast a single value for the whole array
    fn broadcast(value: Rhs, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let mut pipeline = ArrowComputePipeline::new(gpu_device, Some("broadcast"));
        let arr = Self::broadcast_op(value, len, &mut pipeline);
        pipeline.finish();
        arr
    }

    /// Broadcast a single value for the whole array
    fn broadcast_op(value: Rhs, len: usize, pipeline: &mut ArrowComputePipeline) -> Self;
}
