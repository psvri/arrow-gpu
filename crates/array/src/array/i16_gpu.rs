use crate::{
    kernels::{arithmetic::*, cast::Cast},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, gpu_ops::div_ceil, i32_gpu::Int32ArrayGPU, primitive_array_gpu::*,
    u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
};

const I16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../compute_shaders/i16/utils.wgsl"),
    include_str!("../../compute_shaders/i16/cast_i32.wgsl")
);

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

#[async_trait]
impl Cast<Int32ArrayGPU> for Int16ArrayGPU {
    type Output = Int32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                2,
                I16_CAST_I32_SHADER,
                "cast_i32",
            )
            .await;

        Int32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

impl Into<ArrowArrayGPU> for Int16ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::Int16ArrayGPU(self)
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
    use crate::{
        array::{primitive_array_gpu::test::*, ArrowType},
        kernels::cast::cast_dyn,
    };
    use std::sync::Arc;

    test_broadcast!(test_broadcast_i16, Int16ArrayGPU, 1);

    test_cast_op!(
        test_cast_i16_to_i32,
        Int16ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3],
        cast,
        Int32Type,
        vec![0, 1, 2, 3, -1, -2, -3]
    );
}
