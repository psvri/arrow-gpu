use crate::{
    kernels::{arithmetic::*, trigonometry::Trigonometry},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::{sync::Arc};

use super::{
    f32_gpu::Float32ArrayGPU,
    gpu_ops::{div_ceil, u16_ops::sin_u16, u32_ops::*},
    primitive_array_gpu::*,
    ArrowArrayGPU, GpuDevice,
};

pub type UInt16ArrayGPU = PrimitiveArrayGpu<u16>;

impl UInt16ArrayGPU {
    pub async fn braodcast(value: u16, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = div_ceil(len.try_into().unwrap(), 2);
        let boradcast_value = (value as u32) | ((value as u32) << 16);
        let data = Arc::new(braodcast_u32(&gpu_device, boradcast_value, new_len).await);
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
impl Trigonometry for UInt16ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn sin(&self) -> Self::Output {
        let new_buffer = sin_u16(&self.gpu_device, &self.data).await;

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

impl Into<ArrowArrayGPU> for UInt16ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::UInt16ArrayGPU(self)
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
    use crate::{array::primitive_array_gpu::test::*, kernels::trigonometry::sin_dyn};
    use std::sync::Arc;

    test_broadcast!(test_braodcast_u16, u16, 1);

    test_unary_op_float!(
        test_u16_sin,
        UInt16ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3],
        sin,
        sin_dyn,
        vec![0.0f32.sin(), 1.0f32.sin(), 2.0f32.sin(), 3.0f32.sin()]
    );
}
