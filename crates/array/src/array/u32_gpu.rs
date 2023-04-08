use crate::{kernels::logical::Logical, ArrowErrorGPU};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowType,
    NullBitBufferGpu,
};

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

pub const U32_ARRAY_SHADER: &str = include_str!("../../compute_shaders/u32/array.wgsl");
pub const U32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/u32/broadcast.wgsl");

#[async_trait]
impl Logical<UInt32ArrayGPU> for UInt32ArrayGPU {
    type Output = Self;

    async fn bitwise_and(&self, value: &UInt32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                U32_ARRAY_SHADER,
                "bitwise_and",
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

    async fn bitwise_or(&self, value: &UInt32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                U32_ARRAY_SHADER,
                "bitwise_or",
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

impl From<UInt32ArrayGPU> for ArrowArrayGPU {
    fn from(val: UInt32ArrayGPU) -> Self {
        ArrowArrayGPU::UInt32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for UInt32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::UInt32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into UInt32ArrayGPU",
                x
            ))),
        }
    }
}

impl UInt32ArrayGPU {
    pub async fn create_broadcast_buffer(value: u32, len: u64, gpu_device: &GpuDevice) -> Buffer {
        let scalar_buffer = &gpu_device.create_scalar_buffer(&value);
        gpu_device
            .apply_broadcast_function(
                &scalar_buffer,
                4 * len,
                4,
                U32_BROADCAST_SHADER,
                "broadcast",
            )
            .await
    }

    pub async fn broadcast(value: u32, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let data = Arc::new(Self::create_broadcast_buffer(value, len as u64, &gpu_device).await);
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

impl ArrowArray for UInt32ArrayGPU {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> ArrowType {
        ArrowType::UInt32Type
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
    use crate::{
        array::primitive_array_gpu::test::*,
        kernels::logical::{bitwise_and_dyn, bitwise_or_dyn},
    };

    test_broadcast!(test_broadcast_u32, UInt32ArrayGPU, 1);

    test_binary_op!(
        test_bitwise_and_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 100, 260, 450, 0, 1, 100, 260, 450],
        vec![0, 1, 100, 260, 450, !0, !1, !100, !260, !450],
        bitwise_and,
        bitwise_and_dyn,
        vec![0, 1, 100, 260, 450, 0, 0, 0, 0, 0]
    );

    test_binary_op!(
        test_bitwise_and_u32_or_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 100, 260, 450, 0, 1, 100, 260, 450],
        vec![0, 1, 100, 260, 450, !0, !1, !100, !260, !450],
        bitwise_or,
        bitwise_or_dyn,
        vec![
            0,
            1,
            100,
            260,
            450,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ]
    );
}
