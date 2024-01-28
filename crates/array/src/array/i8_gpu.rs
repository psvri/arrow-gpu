use crate::ArrowErrorGPU;
use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
    ArrowComputePipeline,
};

pub type Int8ArrayGPU = PrimitiveArrayGpu<i8>;

impl Int8ArrayGPU {
    pub fn broadcast(value: i8, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = len.div_ceil(4);
        let broadcast_value = (value as u32)
            | ((value as u32) << 8)
            | ((value as u32) << 16)
            | ((value as u32) << 24);
        let gpu_buffer =
            UInt32ArrayGPU::create_broadcast_buffer(broadcast_value, new_len as u64, &gpu_device);
        let data = Arc::new(gpu_buffer);
        let null_buffer = None;

        Self {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }

    pub fn broadcast_op(value: i8, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
        let new_len = len.div_ceil(4);
        let broadcast_value = (value as u32)
            | ((value as u32) << 8)
            | ((value as u32) << 16)
            | ((value as u32) << 24);
        let gpu_buffer =
            UInt32ArrayGPU::create_broadcast_buffer_op(broadcast_value, new_len as u64, pipeline);
        let data = Arc::new(gpu_buffer);
        let null_buffer = None;

        Self {
            data,
            gpu_device: pipeline.device.clone(),
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

impl From<Int8ArrayGPU> for ArrowArrayGPU {
    fn from(val: Int8ArrayGPU) -> Self {
        ArrowArrayGPU::Int8ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Int8ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int8ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int8ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_broadcast!(test_broadcast_i8, Int8ArrayGPU, 1);
}
