use crate::{
    kernels::{arithmetic::*, cast::Cast, trigonometry::Trigonometry},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    f32_gpu::Float32ArrayGPU, gpu_device::GpuDevice, gpu_ops::div_ceil, i32_gpu::Int32ArrayGPU,
    primitive_array_gpu::*, u32_gpu::UInt32ArrayGPU, ArrowArrayGPU,
};

const I8_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../compute_shaders/i8/utils.wgsl"),
    include_str!("../../compute_shaders/i8/trigonometry.wgsl")
);

const I8_CAST_I32_SHADER: &str = concat!(
    include_str!("../../compute_shaders/i8/utils.wgsl"),
    include_str!("../../compute_shaders/i8/cast_i32.wgsl")
);

pub type Int8ArrayGPU = PrimitiveArrayGpu<i8>;

impl Int8ArrayGPU {
    pub async fn broadcast(value: i8, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
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
impl Trigonometry for Int8ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn sin(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                I8_TRIGONOMETRY_SHADER,
                "sin_i8",
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
impl Cast<Int32ArrayGPU> for Int8ArrayGPU {
    type Output = Int32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                I8_CAST_I32_SHADER,
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

impl Into<ArrowArrayGPU> for Int8ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::Int8ArrayGPU(self)
    }
}

impl TryFrom<ArrowArrayGPU> for Int8ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int8ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int8ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::primitive_array_gpu::test::*, array::ArrowType, kernels::cast::*,
        kernels::trigonometry::sin_dyn,
    };
    use std::sync::Arc;

    test_broadcast!(test_broadcast_i8, Int8ArrayGPU, 1);

    test_unary_op_float!(
        test_i8_sin,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 5, -7, -8],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            5.0f32.sin(),
            -7.0f32.sin(),
            -8.0f32.sin(),
        ]
    );

    test_cast_op!(
        test_cast_i8_to_i32,
        Int8ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        cast,
        Int32Type,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7]
    );
}
