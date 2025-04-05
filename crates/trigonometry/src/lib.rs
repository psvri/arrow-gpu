use arrow_gpu_array::array::buffer::ArrowGpuBuffer;
use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;

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

/// Trait for hyperbolic operation on each element of the array
pub trait Hyperbolic: ArrayUtils {
    type Output;

    fn sinh(&self) -> Self::Output {
        default_impl!(self, sinh_op);
    }

    /// Compute sinh(x) for each x in array
    fn sinh_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// Helper trait for Arrow arrays that support hyperbolic functions
pub trait HyperbolicType {
    type OutputType;
    const SHADER: &'static str;
    const TYPE_STR: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: ArrowGpuBuffer,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

/// Trait for trigonometry operation on each element of the array
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

    /// Compute cos(x) for each x in array
    fn cos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
    /// Compute sin(x) for each x in array
    fn sin_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
    /// Compute acos(x) for each x in array
    fn acos_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// Helper trait for Arrow arrays that support trigonometry functions
pub trait TrigonometricType {
    type OutputType;
    const SHADER: &'static str;
    const TYPE_STR: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: ArrowGpuBuffer,
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
            new_buffer.into(),
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
    ($([$dyn: ident, $doc: expr, $dyn_op: ident, $array_op: ident, $($arr:ident),* ]),*) => {
        $(
            #[doc=$doc]
            pub fn $dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
                let mut pipeline = ArrowComputePipeline::new(data.get_gpu_device(), None);
                let result = $dyn_op(data, &mut pipeline);
                pipeline.finish();
                result
            }

            #[doc=concat!("Submits a command to the pipeline to ", $doc)]
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
        "Compute sinh(x) for each x in array",
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
        "Compute cos(x) for each x in array",
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
        "Compute sin(x) for each x in array",
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
        "Compute acos(x) for each x in array",
        acos_op_dyn,
        acos_op,
        Float32ArrayGPU
    ]
);
