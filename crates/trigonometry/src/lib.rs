use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;
use wgpu::Buffer;

pub(crate) mod f32_kernel;
pub(crate) mod i16_kernel;
pub(crate) mod i8_kernel;
pub(crate) mod u16_kernel;
pub(crate) mod u8_kernel;

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
    fn acos(&self) -> Self::Output {
        default_impl!(self, acos_op);
    }

    fn cos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
    fn sin_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
    fn acos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
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

    fn acos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output {
        apply_unary_function_op!(self, TrigonometricType, "acos_{}", pipeline);
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
    [
        sinh_dyn,
        sinh_op_dyn,
        sinh_op,
        Float32ArrayGPU,
        UInt16ArrayGPU,
        UInt8ArrayGPU,
        Int16ArrayGPU,
        Int8ArrayGPU
    ],
    [
        cos_dyn,
        cos_op_dyn,
        cos_op,
        Float32ArrayGPU,
        UInt16ArrayGPU,
        UInt8ArrayGPU,
        Int16ArrayGPU,
        Int8ArrayGPU
    ],
    [
        sin_dyn,
        sin_op_dyn,
        sin_op,
        Float32ArrayGPU,
        UInt16ArrayGPU,
        UInt8ArrayGPU,
        Int16ArrayGPU,
        Int8ArrayGPU
    ],
    [
        acos_dyn,
        acos_op_dyn,
        acos_op,
        Float32ArrayGPU
    ]
);
