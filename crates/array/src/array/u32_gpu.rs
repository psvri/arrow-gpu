use super::{primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowType, NullBitBufferGpu};
use crate::gpu_utils::*;
use crate::ArrowErrorGPU;
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

pub const U32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/u32/broadcast.wgsl");

impl From<UInt32ArrayGPU> for ArrowArrayGPU {
    fn from(val: UInt32ArrayGPU) -> Self {
        ArrowArrayGPU::UInt32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for UInt32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::UInt32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into UInt32ArrayGPU",
                x
            ))),
        }
    }
}

impl UInt32ArrayGPU {
    pub fn create_broadcast_buffer(value: u32, len: u64, gpu_device: &GpuDevice) -> Buffer {
        let scalar_buffer = &gpu_device.create_scalar_buffer(&value);
        gpu_device.apply_broadcast_function(
            scalar_buffer,
            4 * len,
            4,
            U32_BROADCAST_SHADER,
            "broadcast",
        )
    }

    pub fn create_broadcast_buffer_op(
        value: u32,
        len: u64,
        pipeline: &mut ArrowComputePipeline,
    ) -> Buffer {
        let scalar_buffer = &pipeline.device.create_scalar_buffer(&value);

        let dispatch_size = len.div_ceil(256) as u32;

        pipeline.apply_broadcast_function(
            scalar_buffer,
            4 * len,
            U32_BROADCAST_SHADER,
            "broadcast",
            dispatch_size,
        )
    }

    pub fn broadcast(value: u32, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let data = Arc::new(Self::create_broadcast_buffer(
            value,
            len as u64,
            &gpu_device,
        ));
        let null_buffer = None;

        Self {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }

    pub fn broadcast_op(value: u32, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
        let scalar_buffer = pipeline.device.create_scalar_buffer(&value);
        let output_buffer_size = 4 * len as u64;
        let dispatch_size = output_buffer_size.div_ceil(4).div_ceil(256);

        let gpu_buffer = pipeline.apply_broadcast_function(
            &scalar_buffer,
            output_buffer_size,
            U32_BROADCAST_SHADER,
            "broadcast",
            dispatch_size as u32,
        );
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

impl ArrowArray for UInt32ArrayGPU {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> ArrowType {
        ArrowType::UInt32Type
    }

    fn get_memory_used(&self) -> u64 {
        self.data.size()
    }

    fn get_gpu_device(&self) -> &GpuDevice {
        &self.gpu_device
    }

    fn get_buffer(&self) -> &Buffer {
        &self.data
    }

    fn get_null_bit_buffer(&self) -> Option<&NullBitBufferGpu> {
        self.null_buffer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_broadcast!(test_broadcast_u32, UInt32ArrayGPU, 1);
}
