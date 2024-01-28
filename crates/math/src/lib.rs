use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrayUtils, ArrowArrayGPU, ArrowComputePipeline, ArrowPrimitiveType, GpuDevice,
    NullBitBufferGpu, PrimitiveArrayGpu,
};
use wgpu::Buffer;

pub(crate) mod f32;

const ABS_ENTRY_POINT: &str = "abs_";
const SQRT_ENTRY_POINT: &str = "sqrt_";
const EXP_ENTRY_POINT: &str = "exp_";
const EXP2_ENTRY_POINT: &str = "exp2_";
const LOG_ENTRY_POINT: &str = "log_";
const LOG2_ENTRY_POINT: &str = "log2_";

macro_rules! default_impl {
    ($self: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, &mut pipeline);
        pipeline.finish();
        return output;
    };
}

pub trait MathUnaryPass: ArrayUtils {
    type OutputType;

    fn abs(&self) -> Self::OutputType {
        default_impl!(self, abs_op);
    }

    fn abs_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
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

pub trait FloatMathUnary: ArrayUtils {
    type OutputType;
    fn sqrt(&self) -> Self::OutputType {
        default_impl!(self, sqrt_op);
    }
    fn exp(&self) -> Self::OutputType {
        default_impl!(self, exp_op);
    }
    fn exp2(&self) -> Self::OutputType {
        default_impl!(self, exp2_op);
    }
    fn log(&self) -> Self::OutputType {
        default_impl!(self, log_op);
    }
    fn log2(&self) -> Self::OutputType {
        default_impl!(self, log2_op);
    }

    fn sqrt_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
    fn exp_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
    fn exp2_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
    fn log_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
    fn log2_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
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

macro_rules! apply_unary_function_op {
    ($self: ident, $trait_name:ident, $entry_point: ident, $pipeline: ident) => {
        let dispatch_size = $self
            .data
            .size()
            .div_ceil(<T as ArrowPrimitiveType>::ITEM_SIZE)
            .div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_unary_function(
            &$self.data,
            $self.data.size() * <T as $trait_name>::BUFFER_SIZE_MULTIPLIER,
            T::SHADER,
            $entry_point,
            dispatch_size,
        );
        let new_null_buffer = NullBitBufferGpu::clone_null_bit_buffer_pass(
            &$self.null_buffer,
            &mut $pipeline.encoder,
        );

        return <T as $trait_name>::create_new(
            Arc::new(new_buffer),
            $self.gpu_device.clone(),
            $self.len,
            new_null_buffer,
        );
    };
}

impl<T: MathUnaryType + ArrowPrimitiveType> MathUnaryPass for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn abs_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, MathUnaryType, ABS_ENTRY_POINT, pipeline);
    }
}

impl<T: FloatMathUnaryType + ArrowPrimitiveType> FloatMathUnary for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn sqrt_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, SQRT_ENTRY_POINT, pipeline);
    }

    fn exp_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, EXP_ENTRY_POINT, pipeline);
    }

    fn exp2_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, EXP2_ENTRY_POINT, pipeline);
    }

    fn log_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, LOG_ENTRY_POINT, pipeline);
    }

    fn log2_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, LOG2_ENTRY_POINT, pipeline);
    }
}

//TODO replace the below dyn fns with macros

pub fn abs_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.abs().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn abs_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.abs_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn sqrt_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sqrt().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn sqrt_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sqrt_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp2().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn exp2_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.exp2_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log2_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log2().into(),
        _ => panic!("Operation not supported"),
    }
}

pub fn log2_op_dyn(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.log2_op(pipeline).into(),
        _ => panic!("Operation not supported"),
    }
}
