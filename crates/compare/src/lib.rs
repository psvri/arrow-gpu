use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrayUtils, ArrowArrayGPU, ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu,
    PrimitiveArrayGpu,
};
use arrow_gpu_array::gpu_utils::*;

pub(crate) mod f32;
pub(crate) mod i16;
pub(crate) mod i32;
pub(crate) mod i8;
pub(crate) mod u16;
pub(crate) mod u32;
pub(crate) mod u8;

const GT_ENTRY_POINT: &str = "gt";
const GTEQ_ENTRY_POINT: &str = "gteq";
const LT_ENTRY_POINT: &str = "lt";
const LTEQ_ENTRY_POINT: &str = "lteq";
const EQ_ENTRY_POINT: &str = "eq";
const MIN_ENTRY_POINT: &str = "min_";
const MAX_ENTRY_POINT: &str = "max_";

macro_rules! default_impl {
    ($self: ident, $operand: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, $operand, &mut pipeline);
        pipeline.finish();
        return output;
    };
}

/// Helper trait for Arrow arrays that support comparison operation
pub trait CompareType {
    const COMPARE_SHADER: &'static str;
    const MIN_MAX_SHADER: &'static str;
}

/// Trait for comparing ArrowArrays
pub trait Compare: ArrayUtils {
    fn gt(&self, operand: &Self) -> BooleanArrayGPU {
        default_impl!(self, operand, gt_op);
    }
    fn gteq(&self, operand: &Self) -> BooleanArrayGPU {
        default_impl!(self, operand, gteq_op);
    }
    fn lt(&self, operand: &Self) -> BooleanArrayGPU {
        default_impl!(self, operand, lt_op);
    }
    fn lteq(&self, operand: &Self) -> BooleanArrayGPU {
        default_impl!(self, operand, lteq_op);
    }
    fn eq(&self, operand: &Self) -> BooleanArrayGPU {
        default_impl!(self, operand, eq_op);
    }

    /// Returns a boolean array indicating where `self` > `operand`.
    fn gt_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU;
    /// Returns a boolean array indicating where `self` >= `operand`.
    fn gteq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU;
    /// Returns a boolean array indicating where `self` < `operand`.
    fn lt_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU;
    /// Returns a boolean array indicating where `self` <= `operand`.
    fn lteq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU;
    /// Returns a boolean array indicating where `self` == `operand`.
    fn eq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU;
}

/// Trait for finding min and max across each element in ArrowArrays
pub trait MinMax: ArrayUtils + Sized {
    fn max(&self, operand: &Self) -> Self {
        default_impl!(self, operand, max_op);
    }
    fn min(&self, operand: &Self) -> Self {
        default_impl!(self, operand, min_op);
    }

    /// Returns an array where each element is max(self, operand)
    fn max_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self;
    /// Returns an array where each element is min(self, operand)
    fn min_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self;
}

macro_rules! apply_function {
    ($self: ident, $operand:ident, $entry_point: ident, $pipeline: ident) => {
        let dispatch_size = $self.data.size().div_ceil(T::ITEM_SIZE).div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_binary_function(
            &$self.data,
            &$operand.data,
            $self.data.size(),
            T::COMPARE_SHADER,
            $entry_point,
            dispatch_size,
        );

        let null_buffer = NullBitBufferGpu::merge_null_bit_buffer_op(
            &$self.null_buffer,
            &$operand.null_buffer,
            $pipeline,
        );

        return BooleanArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            null_buffer,
        }
    };
}

macro_rules! apply_function_min_max {
    ($self: ident, $operand:ident, $entry_point: ident, $pipeline: ident) => {
        let dispatch_size = $self.data.size().div_ceil(T::ITEM_SIZE).div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_binary_function(
            &$self.data,
            &$operand.data,
            $self.data.size(),
            T::MIN_MAX_SHADER,
            $entry_point,
            dispatch_size,
        );

        let null_buffer = NullBitBufferGpu::merge_null_bit_buffer_op(
            &$self.null_buffer,
            &$operand.null_buffer,
            $pipeline,
        );

        return Self {
            data: new_buffer.into(),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            phantom: std::marker::PhantomData,
            null_buffer,
        }
    };
}

impl<T: CompareType + ArrowPrimitiveType> Compare for PrimitiveArrayGpu<T> {
    fn gt_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
        apply_function!(self, operand, GT_ENTRY_POINT, pipeline);
    }

    fn gteq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
        apply_function!(self, operand, GTEQ_ENTRY_POINT, pipeline);
    }

    fn lt_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
        apply_function!(self, operand, LT_ENTRY_POINT, pipeline);
    }

    fn lteq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
        apply_function!(self, operand, LTEQ_ENTRY_POINT, pipeline);
    }

    fn eq_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
        apply_function!(self, operand, EQ_ENTRY_POINT, pipeline);
    }
}

impl<T: CompareType + ArrowPrimitiveType> MinMax for PrimitiveArrayGpu<T> {
    fn max_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_function_min_max!(self, operand, MAX_ENTRY_POINT, pipeline);
    }

    fn min_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_function_min_max!(self, operand, MIN_ENTRY_POINT, pipeline);
    }
}

macro_rules! dyn_fn {
    ($function:ident, $doc: expr, $op_1:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*) => (
        #[doc=$doc]
        pub fn $function(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> BooleanArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, data_2, &mut pipeline);
            pipeline.finish();
            result
        }

        #[doc=concat!("Submits a command to the pipeline to ", $doc)]
        pub fn $function_op(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> BooleanArrayGPU {
            match (data_1, data_2) {
                $((ArrowArrayGPU::$y(arr_1), ArrowArrayGPU::$y(arr_2)) => arr_1.$op_2(arr_2, pipeline).into(),)+
                _ => panic!(
                    "Operation {} not supported for type {:?} {:?}",
                    stringify!($function),
                    data_1.get_dtype(),
                    data_2.get_dtype(),
                ),
            }
        }
    )
}

dyn_fn!(
    gt_dyn,
    "Construct bool array from computing x > y for each pair (x, y) in zip(data_1, data_2)",
    gt,
    gt_op_dyn,
    gt_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

dyn_fn!(
    gteq_dyn,
    "Construct bool array from computing x >= y for each pair (x, y) in zip(data_1, data_2)",
    gteq,
    gteq_op_dyn,
    gteq_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

dyn_fn!(
    lt_dyn,
    "Construct bool array from computing x < y for each pair (x, y) in zip(data_1, data_2)",
    lt,
    lt_op_dyn,
    lt_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

dyn_fn!(
    lteq_dyn,
    "Construct bool array from computing x <= y for each pair (x, y) in zip(data_1, data_2)",
    lteq,
    lteq_op_dyn,
    lteq_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

dyn_fn!(
    eq_dyn,
    "Construct bool array from computing x == y for each pair (x, y) in zip(data_1, data_2)",
    eq,
    eq_op_dyn,
    eq_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

macro_rules! dyn_minmax {
    ($function:ident, $doc: expr, $op_1:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*) => (
        #[doc=$doc]
        pub fn $function(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, data_2, &mut pipeline);
            pipeline.finish();
            result
        }

        #[doc=concat!("Submits a command to the pipeline to ", $doc)]
        pub fn $function_op(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (data_1, data_2) {
                $((ArrowArrayGPU::$y(arr_1), ArrowArrayGPU::$y(arr_2)) => arr_1.$op_2(arr_2, pipeline).into(),)+
                _ => panic!(
                    "Operation {} not supported for type {:?} {:?}",
                    stringify!($function),
                    data_1.get_dtype(),
                    data_2.get_dtype(),
                ),
            }
        }
    )
}

dyn_minmax!(
    max_dyn,
    "Compute max(x, y) for each pair (x, y) in zip(lhs, rhs)",
    max,
    max_op_dyn,
    max_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);

dyn_minmax!(
    min_dyn,
    "Compute min(x, y) for each pair (x, y) in zip(lhs, rhs)",
    min,
    min_op_dyn,
    min_op,
    Float32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    Int32ArrayGPU,
    Int16ArrayGPU,
    Int8ArrayGPU,
    Date32ArrayGPU
);
