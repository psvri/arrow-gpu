use super::{ArrowArrayGPU, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU};
use crate::ArrowErrorGPU;
use crate::gpu_utils::*;
use crate::kernels::broadcast::Broadcast;
use std::sync::Arc;

pub type Int16ArrayGPU = PrimitiveArrayGpu<i16>;

impl Broadcast<i16> for Int16ArrayGPU {
    fn broadcast_op(value: i16, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
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

impl From<Int16ArrayGPU> for ArrowArrayGPU {
    fn from(val: Int16ArrayGPU) -> Self {
        ArrowArrayGPU::Int16ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Int16ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int16ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int16ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_broadcast!(test_broadcast_i16, Int16ArrayGPU, 1);
}
