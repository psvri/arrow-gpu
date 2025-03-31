use arrow_gpu_array::array::{
    ArrayUtils, ArrowArrayGPU, ArrowPrimitiveType, NullBitBufferGpu, PrimitiveArrayGpu,
};
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;
use wgpu::Buffer;

macro_rules! default_impl {
    ($self: ident, $operand: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, $operand, &mut pipeline);
        pipeline.finish();
        return output;
    };
}

/// The addition operator ArrowArray + Scalar
pub trait ArrowScalarAdd<Rhs>: ArrayUtils {
    type Output;

    fn add_scalar(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, add_scalar_op);
    }

    /// Adds scalar to self
    fn add_scalar_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The subtract operator ArrowArray - Scalar
pub trait ArrowScalarSub<Rhs>: ArrayUtils {
    type Output;

    fn sub_scalar(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, sub_scalar_op);
    }

    /// Subtracts scalar from self
    fn sub_scalar_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The multiplication operator ArrowArray * Scalar
pub trait ArrowScalarMul<Rhs>: ArrayUtils {
    type Output;

    fn mul_scalar(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, mul_scalar_op);
    }

    /// Multiplies scalar with self
    fn mul_scalar_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The division operator ArrowArray / Scalar
pub trait ArrowScalarDiv<Rhs>: ArrayUtils {
    type Output;

    fn div_scalar(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, div_scalar_op);
    }

    /// Divides self with scalar
    fn div_scalar_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The remainder operator ArrowArray % Scalar
pub trait ArrowScalarRem<Rhs>: ArrayUtils {
    type Output;

    fn rem_scalar(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, rem_scalar_op);
    }

    /// Gives remainder of self with scalar
    fn rem_scalar_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

macro_rules! dyn_fn {
    ($function:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*,$([$x: ident, $z: ident]),*) => {
        pub fn $function(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, data_2, &mut pipeline);
            pipeline.finish();
            result
        }

        pub fn $function_op(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (data_1, data_2) {
                $((ArrowArrayGPU::$y(arr_1), ArrowArrayGPU::$y(arr_2)) => arr_1.$op_2(arr_2, pipeline).into(),)+
                $((ArrowArrayGPU::$x(arr_1), ArrowArrayGPU::$z(arr_2)) => arr_1.$op_2(arr_2, pipeline).into(),)*
                _ => panic!(
                    "Operation {} not supported for type {:?} {:?}",
                    stringify!($function),
                    data_1.get_dtype(),
                    data_2.get_dtype(),
                ),
            }
        }
    };
    ($([$dyn: ident, $dyn_op: ident, $array_op: ident, $scalar_op: ident]),*) => {
        $(
            pub fn $dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
                let mut pipeline = ArrowComputePipeline::new(input1.get_gpu_device(), None);
                let result = $dyn_op(input1, input2, &mut pipeline);
                pipeline.finish();
                result
            }

            pub fn $dyn_op(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
                match (input1.len(), input2.len()) {
                    (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => $array_op(input1, input2, pipeline),
                    (_, 1) => $scalar_op(input1, input2, pipeline),
                    (1, _) => $scalar_op(input2, input1, pipeline),
                    _ => unreachable!(),
                }
            }
        )+
    }
}

dyn_fn!(
    add_scalar_dyn,
    add_scalar_op_dyn,
    add_scalar_op,
    Float32ArrayGPU,
    Int32ArrayGPU,
    Date32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
);

dyn_fn!(
    sub_scalar_dyn,
    sub_scalar_op_dyn,
    sub_scalar_op,
    Float32ArrayGPU,
    Int32ArrayGPU,
    UInt32ArrayGPU,
);

dyn_fn!(
    mul_scalar_dyn,
    mul_scalar_op_dyn,
    mul_scalar_op,
    Float32ArrayGPU,
    Int32ArrayGPU,
    UInt32ArrayGPU,
);

dyn_fn!(
    div_scalar_dyn,
    div_scalar_op_dyn,
    div_scalar_op,
    Float32ArrayGPU,
    Int32ArrayGPU,
    UInt32ArrayGPU,
);

dyn_fn!(
    rem_scalar_dyn,
    rem_scalar_op_dyn,
    rem_scalar_op,
    Float32ArrayGPU,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    Date32ArrayGPU,
    [Int32ArrayGPU, Date32ArrayGPU],
    [Date32ArrayGPU, Int32ArrayGPU]
);

/// The addition operator ArrowArray + ArrowArray
pub trait ArrowAdd<Rhs>: ArrayUtils {
    type Output;

    fn add(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, add_op);
    }

    /// Adds array to self
    fn add_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The subtract operator ArrowArray - ArrowArray
pub trait ArrowSub<Rhs>: ArrayUtils {
    type Output;

    fn sub(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, sub_op);
    }

    /// Subtracts array from self
    fn sub_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The multiplication operator ArrowArray * ArrowArray
pub trait ArrowMul<Rhs>: ArrayUtils {
    type Output;

    fn mul(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, mul_op);
    }

    /// Multiplies array with self
    fn mul_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

/// The division operator ArrowArray / ArrowArray
pub trait ArrowDiv<Rhs>: ArrayUtils {
    type Output;

    fn div(&self, value: &Rhs) -> Self::Output {
        default_impl!(self, value, div_op);
    }

    /// Divides self with array
    fn div_op(&self, value: &Rhs, pipeline: &mut ArrowComputePipeline) -> Self::Output;
}

dyn_fn!(
    add_array_dyn,
    add_array_op_dyn,
    add_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    Int32ArrayGPU,
    Date32ArrayGPU,
    [Int32ArrayGPU, Date32ArrayGPU],
    [Date32ArrayGPU, Int32ArrayGPU]
);

dyn_fn!(sub_array_dyn, sub_array_op_dyn, sub_op, Float32ArrayGPU,);

dyn_fn!(mul_array_dyn, mul_array_op_dyn, mul_op, Float32ArrayGPU,);

dyn_fn!(div_array_dyn, div_array_op_dyn, div_op, Float32ArrayGPU,);

dyn_fn!(
    [add_dyn, add_op_dyn, add_array_op_dyn, add_scalar_op_dyn],
    [sub_dyn, sub_op_dyn, sub_array_op_dyn, sub_scalar_op_dyn],
    [mul_dyn, mul_op_dyn, mul_array_op_dyn, mul_scalar_op_dyn],
    [div_dyn, div_op_dyn, div_array_op_dyn, div_scalar_op_dyn]
);

/// The negation operator -ArrowArray
pub trait Neg: ArrayUtils {
    type OutputType;
    fn neg(&self) -> Self::OutputType {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), None);
        let output = self.neg_op(&mut pipeline);
        pipeline.finish();
        output
    }

    fn neg_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType;
}

/// Helper trait for Arrow arrays that support negation operation
pub trait NegUnaryType {
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

impl<T: NegUnaryType + ArrowPrimitiveType> Neg for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn neg_op(&self, pipeline: &mut ArrowComputePipeline) -> Self::OutputType {
        let dispatch_size = self.data.size().div_ceil(T::ITEM_SIZE).div_ceil(256) as u32;

        let new_buffer = pipeline.apply_unary_function(
            &self.data,
            self.data.size() * <T as NegUnaryType>::BUFFER_SIZE_MULTIPLIER,
            T::SHADER,
            "neg",
            dispatch_size,
        );
        let new_null_buffer =
            NullBitBufferGpu::clone_null_bit_buffer_pass(&self.null_buffer, &mut pipeline.encoder);

        <T as NegUnaryType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            new_null_buffer,
        )
    }
}

macro_rules! dyn_neg {
    ($function:ident, $function_op:ident, $( $y:ident ),*) => {
        pub fn $function(data_1: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, &mut pipeline);
            pipeline.finish();
            result
        }

        pub fn $function_op(data_1: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (data_1) {
                $(ArrowArrayGPU::$y(arr_1) => arr_1.neg_op(pipeline).into(),)+
                _ => panic!(
                    "Operation {} not supported for type {:?}",
                    stringify!($function),
                    data_1.get_dtype(),
                ),
            }
        }
    };
}

dyn_neg!(neg_dyn, neg_op_dyn, Float32ArrayGPU);
