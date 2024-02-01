use super::{
    primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowComputePipeline, ArrowPrimitiveType,
    ArrowType, NullBitBufferGpu,
};
use crate::gpu_utils::*;
use crate::{kernels::broadcast::Broadcast, ArrowErrorGPU};
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

const I32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/i32/broadcast.wgsl");

pub type Int32ArrayGPU = PrimitiveArrayGpu<i32>;

impl From<Int32ArrayGPU> for ArrowArrayGPU {
    fn from(val: Int32ArrayGPU) -> Self {
        ArrowArrayGPU::Int32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Int32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int32ArrayGPU",
                x
            ))),
        }
    }
}

impl<T: ArrowPrimitiveType<NativeType = i32>> Broadcast<i32> for PrimitiveArrayGpu<T> {
    type Output = PrimitiveArrayGpu<T>;

    fn broadcast(value: i32, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output {
        let scalar_buffer = gpu_device.create_scalar_buffer(&value);
        let gpu_buffer = gpu_device.apply_broadcast_function(
            &scalar_buffer,
            4 * len as u64,
            4,
            I32_BROADCAST_SHADER,
            "broadcast",
        );
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
}

impl<T: ArrowPrimitiveType<NativeType = i32>> PrimitiveArrayGpu<T> {
    pub fn broadcast_op(value: i32, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
        let scalar_buffer = pipeline.device.create_scalar_buffer(&value);
        let output_buffer_size = 4 * len as u64;
        let dispatch_size = output_buffer_size.div_ceil(4).div_ceil(256);

        let gpu_buffer = pipeline.apply_broadcast_function(
            &scalar_buffer,
            output_buffer_size,
            I32_BROADCAST_SHADER,
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

impl ArrowArray for Int32ArrayGPU {
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

    test_broadcast!(test_broadcast_i32, Int32ArrayGPU, 1);
}
