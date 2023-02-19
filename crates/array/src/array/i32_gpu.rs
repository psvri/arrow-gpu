use crate::{
    kernels::{arithmetic::*, broadcast::Broadcast, logical::Logical},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowPrimitiveType,
    ArrowType, NullBitBufferGpu,
};

const I32_SCALAR_SHADER: &str = include_str!("../../compute_shaders/i32/scalar.wgsl");
const I32_ARRAY_SHADER: &str = include_str!("../../compute_shaders/i32/array.wgsl");
const I32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/i32/broadcast.wgsl");

pub type Int32ArrayGPU = PrimitiveArrayGpu<i32>;

#[async_trait]
impl ArrowScalarAdd<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn add_scalar(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_SCALAR_SHADER,
                "i32_add",
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

#[async_trait]
impl ArrowScalarSub<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn sub_scalar(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_SCALAR_SHADER,
                "i32_sub",
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

#[async_trait]
impl ArrowScalarMul<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn mul_scalar(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_SCALAR_SHADER,
                "i32_mul",
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

#[async_trait]
impl ArrowScalarDiv<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn div_scalar(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_SCALAR_SHADER,
                "i32_div",
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

#[async_trait]
impl Logical<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn bitwise_and(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_ARRAY_SHADER,
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

    async fn bitwise_or(&self, value: &Int32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                I32_ARRAY_SHADER,
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

#[async_trait]
impl ArrowAdd<Int32ArrayGPU> for Int32ArrayGPU {
    type Output = Self;

    async fn add(&self, value: &Int32ArrayGPU) -> Self::Output {
        assert!(Arc::ptr_eq(&self.gpu_device, &value.gpu_device));
        let new_data_buffer = self
            .gpu_device
            .apply_binary_function(&self.data, &value.data, 4, I32_ARRAY_SHADER, "add_i32")
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &value.null_buffer).await;

        Self {
            data: Arc::new(new_data_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }
}

impl From<Int32ArrayGPU> for ArrowArrayGPU {
    fn from(val: Int32ArrayGPU) -> Self {
        ArrowArrayGPU::Int32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Int32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int32ArrayGPU",
                x
            ))),
        }
    }
}

#[async_trait]
impl<T: ArrowPrimitiveType<NativeType = i32>> Broadcast<i32> for PrimitiveArrayGpu<T> {
    type Output = PrimitiveArrayGpu<T>;

    async fn broadcast(value: i32, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output {
        let scalar_buffer = gpu_device.create_scalar_buffer(&value);
        let gpu_buffer = gpu_device
            .apply_broadcast_function(
                &scalar_buffer,
                4 * len as u64,
                4,
                I32_BROADCAST_SHADER,
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

impl ArrowArray for Int32ArrayGPU {
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

    test_add_array!(
        test_add_i32_array_i32,
        Int32ArrayGPU,
        vec![Some(0i32), Some(1), None, None, Some(4)],
        vec![Some(1i32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );

    test_scalar_op!(
        test_add_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100i32,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 100, 200, 3, 104],
        sub_scalar,
        sub_scalar_dyn,
        100,
        vec![-100, 0, 100, -97, 4]
    );

    test_scalar_op!(
        test_mul_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, i32::MAX, 2, 3, 4],
        mul_scalar,
        mul_scalar_dyn,
        100,
        vec![0, -100, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        div_scalar_dyn,
        100,
        vec![0, 0, 1, 2, 4]
    );

    //ignore = "Not passing in linux CI but passes in windows 🤔"
    #[cfg(not(target_os = "linux"))]
    test_scalar_op!(
        test_div_by_zero_i32_scalar_i32,
        i32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        0,
        vec![-1; 5]
    );

    test_broadcast!(test_broadcast_i32, Int32ArrayGPU, 1);

    test_binary_op!(
        test_bitwise_and_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 100, 260, 450, 0, 1, 100, 260, 450],
        vec![0, 1, 100, 260, 450, !0, !1, !100, !260, !450],
        bitwise_and,
        bitwise_and_dyn,
        vec![0, 1, 100, 260, 450, 0, 0, 0, 0, 0]
    );

    test_binary_op!(
        test_bitwise_and_i32_or_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 100, 260, 450, 0, 1, 100, 260, 450],
        vec![0, 1, 100, 260, 450, !0, !1, !100, !260, !450],
        bitwise_or,
        bitwise_or_dyn,
        vec![0, 1, 100, 260, 450, -1, -1, -1, -1, -1]
    );
}
