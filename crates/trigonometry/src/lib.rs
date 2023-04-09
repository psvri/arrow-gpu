pub mod f32_kernel;
pub mod i16_kernel;
pub mod i8_kernel;
pub mod u16_kernel;
pub mod u8_kernel;
use async_trait::async_trait;
use std::sync::Arc;
use wgpu::Buffer;

use arrow_gpu_array::array::*;

pub use self::f32_kernel::*;

#[async_trait]
pub trait Hyperbolic {
    type Output;

    async fn sinh(&self) -> Self::Output;
}

pub trait HyperbolicType {
    type OutputType;
    const SHADER: &'static str;
    const TYPE_STR: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

#[async_trait]
pub trait Trigonometric {
    type Output;

    async fn cos(&self) -> Self::Output;
    async fn sin(&self) -> Self::Output;
}

pub trait TrigonometricType {
    type OutputType;
    const SHADER: &'static str;
    const TYPE_STR: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

#[async_trait]
impl<T: HyperbolicType + ArrowPrimitiveType> Hyperbolic for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    async fn sinh(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * <T as HyperbolicType>::BUFFER_SIZE_MULTIPLIER,
                <T as ArrowPrimitiveType>::ITEM_SIZE as u64,
                T::SHADER,
                format!("sinh_{}", T::TYPE_STR).as_str(),
            )
            .await;

        <T as HyperbolicType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        )
    }
}

#[async_trait]
impl<T: TrigonometricType + ArrowPrimitiveType> Trigonometric for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    async fn cos(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * <T as TrigonometricType>::BUFFER_SIZE_MULTIPLIER,
                <T as ArrowPrimitiveType>::ITEM_SIZE as u64,
                T::SHADER,
                format!("cos_{}", T::TYPE_STR).as_str(),
            )
            .await;

        <T as TrigonometricType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        )
    }

    async fn sin(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * <T as TrigonometricType>::BUFFER_SIZE_MULTIPLIER,
                <T as ArrowPrimitiveType>::ITEM_SIZE as u64,
                T::SHADER,
                format!("sin_{}", T::TYPE_STR).as_str(),
            )
            .await;

        <T as TrigonometricType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        )
    }
}

pub async fn sinh_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sinh().await.into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sinh().await.into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sinh().await.into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sinh().await.into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sinh().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn cos_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.cos().await.into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.cos().await.into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.cos().await.into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.cos().await.into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.cos().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn sin_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sin().await.into(),
        _ => panic!("Operation not supported"),
    }
}
