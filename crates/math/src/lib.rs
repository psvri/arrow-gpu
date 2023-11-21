use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, GpuDevice, NullBitBufferGpu, PrimitiveArrayGpu,
};
use async_trait::async_trait;
use wgpu::Buffer;

pub(crate) mod f32;

const ABS_ENTRY_POINT: &str = "abs_";
const SQRT_ENTRY_POINT: &str = "sqrt_";
const EXP_ENTRY_POINT: &str = "exp_";
const EXP2_ENTRY_POINT: &str = "exp2_";
const LOG_ENTRY_POINT: &str = "log_";
const LOG2_ENTRY_POINT: &str = "log2_";

#[async_trait]
pub trait MathUnary {
    type OutputType;
    async fn abs(&self) -> Self::OutputType;
}

pub trait MathUnaryType {
    type OutputType;
    const SHADER: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

#[async_trait]
pub trait FloatMathUnary {
    type OutputType;
    async fn sqrt(&self) -> Self::OutputType;
    async fn exp(&self) -> Self::OutputType;
    async fn exp2(&self) -> Self::OutputType;
    async fn log(&self) -> Self::OutputType;
    async fn log2(&self) -> Self::OutputType;
}

pub trait FloatMathUnaryType {
    type OutputType;
    const SHADER: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

macro_rules! apply_unary_function {
    ($self: ident, $trait_name:ident, $entry_point: ident) => {
        let new_buffer = $self.gpu_device.apply_unary_function(
            &$self.data,
            &$self.data.size() * <T as $trait_name>::BUFFER_SIZE_MULTIPLIER,
            <T as ArrowPrimitiveType>::ITEM_SIZE,
            T::SHADER,
            $entry_point,
        );
        let new_null_buffer = NullBitBufferGpu::clone_null_bit_buffer(&$self.null_buffer);

        return <T as $trait_name>::create_new(
            Arc::new(new_buffer.await),
            $self.gpu_device.clone(),
            $self.len,
            new_null_buffer.await,
        );
    };
}

#[async_trait]
impl<T: MathUnaryType + ArrowPrimitiveType> MathUnary for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    async fn abs(&self) -> Self::OutputType {
        apply_unary_function!(self, MathUnaryType, ABS_ENTRY_POINT);
    }
}

#[async_trait]
impl<T: FloatMathUnaryType + ArrowPrimitiveType> FloatMathUnary for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    async fn sqrt(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, SQRT_ENTRY_POINT);
    }

    async fn exp(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, EXP_ENTRY_POINT);
    }

    async fn exp2(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, EXP2_ENTRY_POINT);
    }

    async fn log(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, LOG_ENTRY_POINT);
    }

    async fn log2(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, LOG2_ENTRY_POINT);
    }
}

pub async fn abs_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.abs().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn sqrt_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sqrt().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn exp_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn exp2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp2().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn log_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log().await.into(),
        _ => panic!("Operation not supported"),
    }
}

pub async fn log2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log2().await.into(),
        _ => panic!("Operation not supported"),
    }
}
