use crate::{
    kernels::{arithmetic::*, trigonometry::Trigonometry},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    f32_gpu::Float32ArrayGPU,
    gpu_device::GpuDevice,
    gpu_ops::{div_ceil},
    primitive_array_gpu::*,
    u32_gpu::UInt32ArrayGPU,
    ArrowArrayGPU,
};

const U8_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../compute_shaders/u8/utils.wgsl"),
    include_str!("../../compute_shaders/u8/trigonometry.wgsl")
);

pub type UInt8ArrayGPU = PrimitiveArrayGpu<u8>;

impl UInt8ArrayGPU {
    pub async fn broadcast(value: u8, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let new_len = div_ceil(len.try_into().unwrap(), 4);
        let broadcast_value = (value as u32)
            | ((value as u32) << 8)
            | ((value as u32) << 16)
            | ((value as u32) << 24);
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
impl Trigonometry for UInt8ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn sin(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                U8_TRIGONOMETRY_SHADER,
                "sin_u8",
            )
            .await;

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

impl Into<ArrowArrayGPU> for UInt8ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::UInt8ArrayGPU(self)
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
    use crate::{array::primitive_array_gpu::test::*, kernels::trigonometry::sin_dyn};
    use std::sync::Arc;

    test_broadcast!(test_broadcast_u8, UInt8ArrayGPU, 1);

    test_unary_op_float!(
        test_u8_sin,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 5],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            5.0f32.sin()
        ]
    );
}
