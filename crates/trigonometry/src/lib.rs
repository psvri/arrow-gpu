pub(crate) mod f32_kernel;
pub(crate) mod i16_kernel;
pub(crate) mod i8_kernel;
pub(crate) mod u16_kernel;
pub(crate) mod u8_kernel;
use std::sync::Arc;
use wgpu::Buffer;

use arrow_gpu_array::array::*;

macro_rules! default_impl {
    ($self: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, &mut pipeline);
        pipeline.finish();
        return output;
    };
}

pub trait Hyperbolic: ArrayUtils {
    type Output;

    fn sinh(&self) -> Self::Output {
        default_impl!(self, sinh_op);
    }

    fn sinh_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
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

pub trait Trigonometric: ArrayUtils {
    type Output;

    fn cos(&self) -> Self::Output {
        default_impl!(self, cos_op);
    }
    fn sin(&self) -> Self::Output {
        default_impl!(self, sin_op);
    }

    fn cos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
    fn sin_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
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

macro_rules! apply_unary_function_op {
    ($self: ident, $trait_name:ident, $entry_point: literal, $pipeline: ident) => {
        let dispatch_size = $self
            .data
            .size()
            .div_ceil(<T as ArrowPrimitiveType>::ITEM_SIZE)
            .div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_unary_function(
            &$self.data,
            $self.data.size() * <T as $trait_name>::BUFFER_SIZE_MULTIPLIER,
            T::SHADER,
            format!($entry_point, T::TYPE_STR).as_str(),
            dispatch_size,
        );

        let null_buffer = NullBitBufferGpu::clone_null_bit_buffer_pass(
            &$self.null_buffer,
            &mut $pipeline.encoder,
        );

        return <T as $trait_name>::create_new(
            Arc::new(new_buffer),
            $self.gpu_device.clone(),
            $self.len,
            null_buffer,
        );
    };
}

impl<T: HyperbolicType + ArrowPrimitiveType> Hyperbolic for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    fn sinh_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output {
        apply_unary_function_op!(self, HyperbolicType, "sinh_{}", pipeline);
    }
}

impl<T: TrigonometricType + ArrowPrimitiveType> Trigonometric for PrimitiveArrayGpu<T> {
    type Output = T::OutputType;

    fn cos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output {
        apply_unary_function_op!(self, TrigonometricType, "cos_{}", pipeline);
    }

    fn sin_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output {
        apply_unary_function_op!(self, TrigonometricType, "sin_{}", pipeline);
    }
}

//TODO replace the below fns with macros

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

pub fn sinh_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sinh_op(pipeline).into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sinh_op(pipeline).into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sinh_op(pipeline).into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sinh_op(pipeline).into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sinh_op(pipeline).into(),
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

pub fn cos_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.cos_op(pipeline).into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.cos_op(pipeline).into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.cos_op(pipeline).into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.cos_op(pipeline).into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.cos_op(pipeline).into(),
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

pub fn sin_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sin_op(pipeline).into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sin_op(pipeline).into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sin_op(pipeline).into(),
        ArrowArrayGPU::Int16ArrayGPU(arr) => arr.sin_op(pipeline).into(),
        ArrowArrayGPU::Int8ArrayGPU(arr) => arr.sin_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}
