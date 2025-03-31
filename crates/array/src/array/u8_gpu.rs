use super::{ArrowArrayGPU, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU};
use crate::ArrowErrorGPU;
use crate::gpu_utils::*;
use crate::kernels::broadcast::Broadcast;
use std::sync::Arc;

/// UInt8 arrow array in gpu
pub type UInt8ArrayGPU = PrimitiveArrayGpu<u8>;

impl Broadcast<u8> for UInt8ArrayGPU {
    fn broadcast_op(value: u8, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
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

impl From<UInt8ArrayGPU> for ArrowArrayGPU {
    fn from(val: UInt8ArrayGPU) -> ArrowArrayGPU {
        ArrowArrayGPU::UInt8ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for UInt8ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::UInt8ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into UInt8ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_broadcast!(test_broadcast_u8, UInt8ArrayGPU, 1);
}
