use crate::{
    kernels::{arithmetic::*, cast::Cast, trigonometry::Trigonometry},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    f32_gpu::Float32ArrayGPU,
    gpu_ops::{div_ceil, i16_ops::*, u32_ops::*},
    i32_gpu::Int32ArrayGPU,
    primitive_array_gpu::*,
    ArrowArrayGPU, GpuDevice,
};

pub type Int16ArrayGPU = PrimitiveArrayGpu<i16>;

impl Int16ArrayGPU {
    pub async fn broadcast(value: i16, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = div_ceil(len.try_into().unwrap(), 2);
        let broadcast_value = (value as u32) | ((value as u32) << 16);
        let data = Arc::new(broadcast_u32(&gpu_device, broadcast_value, new_len).await);
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
impl Trigonometry for Int16ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn sin(&self) -> Self::Output {
        let new_buffer = sin_i16(&self.gpu_device, &self.data).await;

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

#[async_trait]
impl Cast<Int32ArrayGPU> for Int16ArrayGPU {
    type Output = Int32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = cast_i32(&self.gpu_device, &self.data).await;

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
        kernels::{cast::cast_dyn, trigonometry::sin_dyn},
    };
    use std::sync::Arc;

    test_broadcast!(test_broadcast_i16, Int16ArrayGPU, 1);

    test_unary_op_float!(
        test_i16_sin,
        Int16ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            -1.0f32.sin(),
            -2.0f32.sin(),
            -3.0f32.sin()
        ]
    );

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
