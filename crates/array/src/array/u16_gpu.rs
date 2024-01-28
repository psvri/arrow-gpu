use crate::ArrowErrorGPU;
use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
    ArrowComputePipeline,
};

pub type UInt16ArrayGPU = PrimitiveArrayGpu<u16>;

impl UInt16ArrayGPU {
    pub fn broadcast(value: u16, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = len.div_ceil(2);
        let broadcast_value = (value as u32) | ((value as u32) << 16);
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

    pub fn broadcast_op(value: u16, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
        let new_len = len.div_ceil(2);
        let broadcast_value = (value as u32) | ((value as u32) << 16);
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

impl From<UInt16ArrayGPU> for ArrowArrayGPU {
    fn from(val: UInt16ArrayGPU) -> Self {
        ArrowArrayGPU::UInt16ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for UInt16ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::UInt16ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into UInt16ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_broadcast!(test_broadcast_u16, UInt16ArrayGPU, 1u16);
}
