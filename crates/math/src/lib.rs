use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrayUtils, ArrowArrayGPU, ArrowPrimitiveType, NullBitBufferGpu, PrimitiveArrayGpu,
};
use arrow_gpu_array::gpu_utils::*;
use wgpu::Buffer;

pub(crate) mod f32;

const ABS_ENTRY_POINT: &str = "abs_";
const SQRT_ENTRY_POINT: &str = "sqrt_";
const CBRT_ENTRY_POINT: &str = "cbrt_";
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
    fn cbrt(&self) -> Self::OutputType {
        default_impl!(self, cbrt_op);
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
    fn cbrt_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
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

    fn cbrt_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        apply_unary_function_op!(self, FloatMathUnaryType, CBRT_ENTRY_POINT, pipeline);
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

macro_rules! dyn_fn {
    ($([$dyn: ident, $dyn_op: ident, $array_op: ident, $($arr:ident),* ]),*) => {
        $(
            pub fn $dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
                let mut pipeline = ArrowComputePipeline::new(data.get_gpu_device(), None);
                let result = $dyn_op(data, &mut pipeline);
                pipeline.finish();
                result
            }

            pub fn $dyn_op(data: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
                match data {
                    $(ArrowArrayGPU::$arr(arr_1) => arr_1.$array_op(pipeline).into(),)*
                    _ => panic!("Operation {} not supported for type {:?}", stringify!($dyn_op), data.get_dtype())
                }
            }
        )+
    }
}

dyn_fn!(
    [abs_dyn, abs_op_dyn, abs_op, Float32ArrayGPU],
    [sqrt_dyn, sqrt_op_dyn, sqrt_op, Float32ArrayGPU],
    [cbrt_dyn, cbrt_op_dyn, cbrt_op, Float32ArrayGPU],
    [exp_dyn, exp_op_dyn, exp_op, Float32ArrayGPU],
    [exp2_dyn, exp2_op_dyn, exp2_op, Float32ArrayGPU],
    [log_dyn, log_op_dyn, log_op, Float32ArrayGPU],
    [log2_dyn, log2_op_dyn, log2_op, Float32ArrayGPU]
);
