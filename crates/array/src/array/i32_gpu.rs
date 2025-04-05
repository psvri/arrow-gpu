use super::buffer::ArrowGpuBuffer;
use super::{
    ArrowArray, ArrowArrayGPU, ArrowComputePipeline, ArrowPrimitiveType, ArrowType,
    NullBitBufferGpu, primitive_array_gpu::*,
};
use crate::gpu_utils::*;
use crate::{ArrowErrorGPU, kernels::broadcast::Broadcast};
use std::any::Any;

const I32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/i32/broadcast.wgsl");

/// Int32 arrow array in gpu
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
    fn broadcast_op(
        value: i32,
        len: usize,
        pipeline: &mut ArrowComputePipeline,
    ) -> PrimitiveArrayGpu<T> {
        let scalar_buffer = pipeline.device.create_scalar_buffer(&value);
        let gpu_buffer = pipeline.apply_broadcast_function(
            &scalar_buffer,
            4 * len as u64,
            I32_BROADCAST_SHADER,
            "broadcast",
            len.div_ceil(256) as u32,
        );
        let data = gpu_buffer.into();
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

    fn get_buffer(&self) -> &ArrowGpuBuffer {
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
