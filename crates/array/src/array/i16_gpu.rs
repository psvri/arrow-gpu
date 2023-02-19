use crate::ArrowErrorGPU;

use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, gpu_ops::div_ceil, primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU,
    ArrowArrayGPU,
};

pub type Int16ArrayGPU = PrimitiveArrayGpu<i16>;

impl Int16ArrayGPU {
    pub async fn broadcast(value: i16, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = div_ceil(len.try_into().unwrap(), 2);
        let broadcast_value = (value as u32) | ((value as u32) << 16);
        let gpu_buffer =
            UInt32ArrayGPU::create_broadcast_buffer(broadcast_value, new_len, &gpu_device).await;
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
    use std::sync::Arc;

    test_broadcast!(test_broadcast_i16, Int16ArrayGPU, 1);
}
