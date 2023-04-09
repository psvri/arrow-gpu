use crate::{kernels::aggregate::ArrowSum, ArrowErrorGPU};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

use super::{
    gpu_device::GpuDevice, gpu_ops::f32_ops::*, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU,
    ArrowType, NullBitBufferGpu,
};

const F32_REDUCTION_SHADER: &str = include_str!("../../compute_shaders/f32/reduction.wgsl");
const F32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/f32/broadcast.wgsl");

pub type Float32ArrayGPU = PrimitiveArrayGpu<f32>;

impl_unary_ops!(ArrowSum, sum, Float32ArrayGPU, f32, sum);

impl Float32ArrayGPU {
    pub async fn broadcast(value: f32, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let scalar_buffer = &gpu_device.create_scalar_buffer(&value);
        let gpu_buffer = gpu_device
            .apply_broadcast_function(
                scalar_buffer,
                4 * len as u64,
                4,
                F32_BROADCAST_SHADER,
                "broadcast",
            )
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

impl From<Float32ArrayGPU> for ArrowArrayGPU {
    fn from(val: Float32ArrayGPU) -> Self {
        ArrowArrayGPU::Float32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Float32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Float32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Float32ArrayGPU",
                x
            ))),
        }
    }
}

impl ArrowArray for Float32ArrayGPU {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> ArrowType {
        ArrowType::Float32Type
    }

    fn get_memory_used(&self) -> u64 {
        self.data.size()
    }

    fn get_gpu_device(&self) -> &GpuDevice {
        &self.gpu_device
    }

    fn get_buffer(&self) -> &Buffer {
        &self.data
    }

    fn get_null_bit_buffer(&self) -> Option<&NullBitBufferGpu> {
        self.null_buffer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    #[ignore = "Not passing in CI but passes in local 🤔"]
    #[tokio::test]
    async fn test_f32_sum() {
        let device = Arc::new(GpuDevice::new().await);
        let gpu_array = Float32ArrayGPU::from_vec(
            &(0..256 * 256)
                .into_iter()
                .map(|_| 1.0)
                .collect::<Vec<f32>>(),
            device.clone(),
        );

        assert_eq!(gpu_array.sum().await, 65536.0);

        // TODO fix this
        let cvec = (0..9_999)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let total = (0..9_999u32).into_iter().sum::<u32>() as f32;
        let gpu_array = Float32ArrayGPU::from_vec(&cvec, device);
        assert_eq!(gpu_array.sum().await, total);
    }

    #[tokio::test]
    async fn test_f32_array_from_optinal_vec() {
        let device = Arc::new(GpuDevice::new().await);
        let gpu_array_1 = Float32ArrayGPU::from_optional_vec(
            &vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
            device.clone(),
        );
        assert_eq!(
            gpu_array_1.raw_values().await.unwrap(),
            vec![0.0, 1.0, 0.0, 0.0, 4.0]
        );
        assert_eq!(
            gpu_array_1.null_buffer.as_ref().unwrap().raw_values().await,
            vec![0b00010011]
        );
        let gpu_array_2 = Float32ArrayGPU::from_optional_vec(
            &vec![Some(1.0), Some(2.0), None, Some(4.0), None],
            device,
        );
        assert_eq!(
            gpu_array_2.raw_values().await.unwrap(),
            vec![1.0, 2.0, 0.0, 4.0, 0.0]
        );
        assert_eq!(
            gpu_array_2.null_buffer.as_ref().unwrap().raw_values().await,
            vec![0b00001011]
        );
        let new_bit_buffer = NullBitBufferGpu::merge_null_bit_buffer(
            &gpu_array_2.null_buffer,
            &gpu_array_1.null_buffer,
        )
        .await;
        assert_eq!(new_bit_buffer.unwrap().raw_values().await, vec![0b00000011]);
    }

    test_broadcast!(test_broadcast_f32, Float32ArrayGPU, 1.0);
}
