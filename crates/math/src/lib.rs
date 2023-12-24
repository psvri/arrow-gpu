use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, GpuDevice, NullBitBufferGpu, PrimitiveArrayGpu,
};
use wgpu::Buffer;

pub(crate) mod f32;

const ABS_ENTRY_POINT: &str = "abs_";
const SQRT_ENTRY_POINT: &str = "sqrt_";
const EXP_ENTRY_POINT: &str = "exp_";
const EXP2_ENTRY_POINT: &str = "exp2_";
const LOG_ENTRY_POINT: &str = "log_";
const LOG2_ENTRY_POINT: &str = "log2_";

pub trait MathUnary {
    type OutputType;
    fn abs(&self) -> Self::OutputType;
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

pub trait FloatMathUnary {
    type OutputType;
    fn sqrt(&self) -> Self::OutputType;
    fn exp(&self) -> Self::OutputType;
    fn exp2(&self) -> Self::OutputType;
    fn log(&self) -> Self::OutputType;
    fn log2(&self) -> Self::OutputType;
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
            Arc::new(new_buffer),
            $self.gpu_device.clone(),
            $self.len,
            new_null_buffer,
        );
    };
}

impl<T: MathUnaryType + ArrowPrimitiveType> MathUnary for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn abs(&self) -> Self::OutputType {
        apply_unary_function!(self, MathUnaryType, ABS_ENTRY_POINT);
    }
}

impl<T: FloatMathUnaryType + ArrowPrimitiveType> FloatMathUnary for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn sqrt(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, SQRT_ENTRY_POINT);
    }

    fn exp(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, EXP_ENTRY_POINT);
    }

    fn exp2(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, EXP2_ENTRY_POINT);
    }

    fn log(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, LOG_ENTRY_POINT);
    }

    fn log2(&self) -> Self::OutputType {
        apply_unary_function!(self, FloatMathUnaryType, LOG2_ENTRY_POINT);
    }
}

pub fn abs_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.abs().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn sqrt_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sqrt().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp2().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log2().into(),
        _ => panic!("Operation not supported"),
    }
}
