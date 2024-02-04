use std::sync::Arc;

use crate::gpu_utils::{ArrowComputePipeline, GpuDevice};

pub trait Broadcast<Rhs>: Sized {
    fn broadcast(value: Rhs, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let mut pipeline = ArrowComputePipeline::new(gpu_device, Some("broadcast"));
        let arr = Self::broadcast_op(value, len, &mut pipeline);
        pipeline.finish();
        arr
    }

    fn broadcast_op(value: Rhs, len: usize, pipeline: &mut ArrowComputePipeline) -> Self;
}
