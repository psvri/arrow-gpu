use crate::ArrowErrorGPU;

use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
};
pub type UInt8ArrayGPU = PrimitiveArrayGpu<u8>;

impl UInt8ArrayGPU {
    pub async fn broadcast(value: u8, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = len.div_ceil(4);
        let broadcast_value = (value as u32)
            | ((value as u32) << 8)
            | ((value as u32) << 16)
            | ((value as u32) << 24);
        let gpu_buffer =
            UInt32ArrayGPU::create_broadcast_buffer(broadcast_value, new_len as u64, &gpu_device)
                .await;
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
