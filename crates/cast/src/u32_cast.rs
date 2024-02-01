use std::sync::Arc;

use arrow_gpu_array::array::{ArrayUtils, Float32ArrayGPU, NullBitBufferGpu, UInt32ArrayGPU};
use arrow_gpu_array::gpu_utils::*;

use crate::BitCast;

impl BitCast<Float32ArrayGPU> for UInt32ArrayGPU {
    fn bitcast_op(&self, pipeline: &mut ArrowComputePipeline) -> Float32ArrayGPU {
        let data = pipeline.clone_buffer(&self.data);
        let null_buffer = NullBitBufferGpu::clone_null_bit_buffer_op(&self.null_buffer, pipeline);
        let data = Arc::new(data);
        Float32ArrayGPU {
            data,
            gpu_device: self.get_gpu_device(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer,
        }
    }
}
