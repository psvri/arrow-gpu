pub mod boolean;
pub mod i16;
pub mod i32;
pub mod i8;
pub mod u16;
pub mod u32;
pub mod u8;

use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

pub(crate) const AND_ENTRY_POINT: &str = "bitwise_and";
pub(crate) const OR_ENTRY_POINT: &str = "bitwise_or";
pub(crate) const XOR_ENTRY_POINT: &str = "bitwise_xor";
pub(crate) const NOT_ENTRY_POINT: &str = "bitwise_not";
pub(crate) const SHIFT_LEFT_ENTRY_POINT: &str = "bitwise_shl";
pub(crate) const SHIFT_RIGHT_ENTRY_POINT: &str = "bitwise_shr";

pub trait LogicalType {
    const SHADER: &'static str;
    const SHIFT_SHADER: &'static str;
    const NOT_SHADER: &'static str;
}

#[async_trait]
pub trait Logical {
    async fn bitwise_and(&self, operand: &Self) -> Self;
    async fn bitwise_or(&self, operand: &Self) -> Self;
    async fn bitwise_xor(&self, operand: &Self) -> Self;
    async fn bitwise_not(&self) -> Self;
    async fn bitwise_shl(&self, operand: &UInt32ArrayGPU) -> Self;
    async fn bitwise_shr(&self, operand: &UInt32ArrayGPU) -> Self;
}

#[async_trait]
impl<T: LogicalType + ArrowPrimitiveType> Logical for PrimitiveArrayGpu<T> {
    async fn bitwise_and(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                T::ITEM_SIZE,
                T::SHADER,
                AND_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_or(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                T::ITEM_SIZE,
                T::SHADER,
                OR_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_xor(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                T::ITEM_SIZE,
                T::SHADER,
                XOR_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_not(&self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size(),
                T::ITEM_SIZE,
                T::NOT_SHADER,
                NOT_ENTRY_POINT,
            )
            .await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }

    async fn bitwise_shl(&self, operand: &UInt32ArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                T::ITEM_SIZE,
                T::SHIFT_SHADER,
                SHIFT_LEFT_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_shr(&self, operand: &UInt32ArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                T::ITEM_SIZE,
                T::SHIFT_SHADER,
                SHIFT_RIGHT_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }
}

pub async fn bitwise_and_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::Int32ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr_1), ArrowArrayGPU::UInt16ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(arr_1), ArrowArrayGPU::Int16ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(arr_1), ArrowArrayGPU::UInt8ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(arr_1), ArrowArrayGPU::Int8ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::BooleanArrayGPU(arr_1), ArrowArrayGPU::BooleanArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_or_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::Int32ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr_1), ArrowArrayGPU::UInt16ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(arr_1), ArrowArrayGPU::Int16ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(arr_1), ArrowArrayGPU::UInt8ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(arr_1), ArrowArrayGPU::Int8ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::BooleanArrayGPU(arr_1), ArrowArrayGPU::BooleanArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_xor_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::Int32ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr_1), ArrowArrayGPU::UInt16ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(arr_1), ArrowArrayGPU::Int16ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(arr_1), ArrowArrayGPU::UInt8ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(arr_1), ArrowArrayGPU::Int8ArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        (ArrowArrayGPU::BooleanArrayGPU(arr_1), ArrowArrayGPU::BooleanArrayGPU(arr_2)) => {
            arr_1.bitwise_xor(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_shl_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shl(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_shr_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_shr(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_not_dyn(data_1: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data_1 {
        ArrowArrayGPU::Int32ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        ArrowArrayGPU::UInt32ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        ArrowArrayGPU::Int16ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        ArrowArrayGPU::Int8ArrayGPU(arr_1) => arr_1.bitwise_not().await.into(),
        _ => panic!("Operation not supported"),
    }
}
