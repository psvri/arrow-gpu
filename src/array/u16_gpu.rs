use crate::{
    kernels::{arithmetic::*, trigonometry::Trigonometry},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    f32_gpu::Float32ArrayGPU, gpu_device::GpuDevice, gpu_ops::div_ceil, primitive_array_gpu::*,
    u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
};

const U16_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../compute_shaders/u16/utils.wgsl"),
    include_str!("../../compute_shaders/u16/trigonometry.wgsl")
);

const U16_SCALAR_SHADER: &str = concat!(
    include_str!("../../compute_shaders/u16/utils.wgsl"),
    include_str!("../../compute_shaders/u16/scalar.wgsl")
);

pub type UInt16ArrayGPU = PrimitiveArrayGpu<u16>;

impl UInt16ArrayGPU {
    pub async fn broadcast(value: u16, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
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
impl Trigonometry for UInt16ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn sin(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                2,
                U16_TRIGONOMETRY_SHADER,
                "sin_u16",
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

#[async_trait]
impl ArrowScalarAdd<UInt16ArrayGPU> for UInt16ArrayGPU {
    type Output = Self;

    async fn add_scalar(&self, value: &UInt16ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                2,
                U16_SCALAR_SHADER,
                "u16_add",
            )
            .await;

        Self {
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

    test_broadcast!(test_broadcast_u16, UInt16ArrayGPU, 1u16);

    test_unary_op_float!(
        test_u16_sin,
        UInt16ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3],
        sin,
        sin_dyn,
        vec![0.0f32.sin(), 1.0f32.sin(), 2.0f32.sin(), 3.0f32.sin()]
    );

    test_scalar_op!(
        test_add_u16_scalar_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100u16,
        vec![100, 101, 102, 103, 104]
    );
}
