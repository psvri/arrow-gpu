pub(crate) mod f32_kernel;
pub(crate) mod i16_kernel;
pub(crate) mod i8_kernel;
pub(crate) mod u16_kernel;
pub(crate) mod u8_kernel;
use std::sync::Arc;
use wgpu::Buffer;

use arrow_gpu_array::array::*;

pub use self::f32_kernel::*;

pub trait Hyperbolic {
    type Output;

    fn sinh(&self) -> Self::Output;
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

pub trait Trigonometric {
    type Output;

    fn cos(&self) -> Self::Output;
    fn sin(&self) -> Self::Output;
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

impl<T: HyperbolicType + ArrowPrimitiveType> Hyperbolic for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    fn sinh(&self) -> Self::Output {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * <T as HyperbolicType>::BUFFER_SIZE_MULTIPLIER,
            <T as ArrowPrimitiveType>::ITEM_SIZE,
            T::SHADER,
            format!("sinh_{}", T::TYPE_STR).as_str(),
        );

        <T as HyperbolicType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        )
    }
}

impl<T: TrigonometricType + ArrowPrimitiveType> Trigonometric for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    fn cos(&self) -> Self::Output {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * <T as TrigonometricType>::BUFFER_SIZE_MULTIPLIER,
            <T as ArrowPrimitiveType>::ITEM_SIZE,
            T::SHADER,
            format!("cos_{}", T::TYPE_STR).as_str(),
        );

        <T as TrigonometricType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        )
    }

    fn sin(&self) -> Self::Output {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * <T as TrigonometricType>::BUFFER_SIZE_MULTIPLIER,
            <T as ArrowPrimitiveType>::ITEM_SIZE,
            T::SHADER,
            format!("sin_{}", T::TYPE_STR).as_str(),
        );

        <T as TrigonometricType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        )
    }
}

pub fn sinh_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sinh().into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sinh().into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sinh().into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sinh().into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sinh().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn cos_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.cos().into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.cos().into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.cos().into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.cos().into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.cos().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn sin_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sin().into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sin().into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sin().into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sin().into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sin().into(),
        _ => panic!("Operation not supported"),
    }
}
