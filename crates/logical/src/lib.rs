pub(crate) mod boolean;
pub(crate) mod i16;
pub(crate) mod i32;
pub(crate) mod i8;
pub(crate) mod u16;
pub(crate) mod u32;
pub(crate) mod u8;

use std::sync::Arc;

use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;

pub(crate) const AND_ENTRY_POINT: &str = "bitwise_and";
pub(crate) const OR_ENTRY_POINT: &str = "bitwise_or";
pub(crate) const XOR_ENTRY_POINT: &str = "bitwise_xor";
pub(crate) const NOT_ENTRY_POINT: &str = "bitwise_not";
pub(crate) const SHIFT_LEFT_ENTRY_POINT: &str = "bitwise_shl";
pub(crate) const SHIFT_RIGHT_ENTRY_POINT: &str = "bitwise_shr";

/// Helper trait for Arrow arrays that support logical operation
pub trait LogicalType {
    const SHADER: &'static str;
    const SHIFT_SHADER: &'static str;
    const NOT_SHADER: &'static str;
}

macro_rules! default_impl {
    ($self: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, &mut pipeline);
        pipeline.finish();
        return output;
    };
    ($self: ident, $operand: ident, $fn: ident) => {
        let mut pipeline = ArrowComputePipeline::new($self.get_gpu_device(), None);
        let output = Self::$fn(&$self, $operand, &mut pipeline);
        pipeline.finish();
        return output;
    };
}

/// Trait for logical operation on each element of the array
pub trait Logical: ArrayUtils + Sized {
    fn bitwise_and(&self, operand: &Self) -> Self {
        default_impl!(self, operand, bitwise_and_op);
    }
    fn bitwise_or(&self, operand: &Self) -> Self {
        default_impl!(self, operand, bitwise_or_op);
    }
    fn bitwise_xor(&self, operand: &Self) -> Self {
        default_impl!(self, operand, bitwise_xor_op);
    }
    fn bitwise_not(&self) -> Self {
        default_impl!(self, bitwise_not_op);
    }
    fn bitwise_shl(&self, operand: &UInt32ArrayGPU) -> Self {
        default_impl!(self, operand, bitwise_shl_op);
    }
    fn bitwise_shr(&self, operand: &UInt32ArrayGPU) -> Self {
        default_impl!(self, operand, bitwise_shr_op);
    }

    /// For each pair (x, y) in zip(self, operand) compute x & y
    fn bitwise_and_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self;
    /// For each pair (x, y) in zip(self, operand) compute x | y
    fn bitwise_or_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self;
    /// For each pair (x, y) in zip(self, operand) compute x ^ y
    fn bitwise_xor_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self;
    /// For each element x in the array computes !x
    fn bitwise_not_op(&self, pipeline: &mut ArrowComputePipeline) -> Self;
    /// For each pair (x, y) in zip(self, operand) compute x << y
    fn bitwise_shl_op(&self, operand: &UInt32ArrayGPU, pipeline: &mut ArrowComputePipeline)
    -> Self;
    /// For each pair (x, y) in zip(self, operand) compute x >> y
    fn bitwise_shr_op(&self, operand: &UInt32ArrayGPU, pipeline: &mut ArrowComputePipeline)
    -> Self;
}

/// Trait for is bit set operations
pub trait LogicalContains {
    /// Check if all bits are set in the array
    fn any(&self) -> bool;
    /// Check if any bit is set in the array
    fn all(&self) -> bool;
}

macro_rules! apply_binary_function_op {
    ($self: ident, $operand: ident, $shader: ident, $entry_point: ident, $pipeline: ident) => {
        let dispatch_size = $self
            .data
            .size()
            .div_ceil(<T as ArrowPrimitiveType>::ITEM_SIZE)
            .div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_binary_function(
            &$self.data,
            &$operand.data,
            $self.data.size(),
            T::$shader,
            $entry_point,
            dispatch_size,
        );
        let null_buffer = NullBitBufferGpu::merge_null_bit_buffer_op(
            &$self.null_buffer,
            &$operand.null_buffer,
            $pipeline,
        );

        return Self {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: $self.len,
            null_buffer,
        };
    };
}

impl<T: LogicalType + ArrowPrimitiveType> Logical for PrimitiveArrayGpu<T> {
    fn bitwise_and_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, AND_ENTRY_POINT, pipeline);
    }

    fn bitwise_or_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, OR_ENTRY_POINT, pipeline);
    }

    fn bitwise_xor_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, XOR_ENTRY_POINT, pipeline);
    }

    fn bitwise_not_op(&self, pipeline: &mut ArrowComputePipeline) -> Self {
        let dispatch_size = self
            .data
            .size()
            .div_ceil(<T as ArrowPrimitiveType>::ITEM_SIZE)
            .div_ceil(256) as u32;

        let new_buffer = pipeline.apply_unary_function(
            &self.data,
            self.data.size(),
            T::NOT_SHADER,
            NOT_ENTRY_POINT,
            dispatch_size,
        );

        let null_buffer =
            NullBitBufferGpu::clone_null_bit_buffer_pass(&self.null_buffer, &mut pipeline.encoder);

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer,
        }
    }

    fn bitwise_shl_op(
        &self,
        operand: &UInt32ArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) -> Self {
        apply_binary_function_op!(
            self,
            operand,
            SHIFT_SHADER,
            SHIFT_LEFT_ENTRY_POINT,
            pipeline
        );
    }

    fn bitwise_shr_op(
        &self,
        operand: &UInt32ArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) -> Self {
        apply_binary_function_op!(
            self,
            operand,
            SHIFT_SHADER,
            SHIFT_RIGHT_ENTRY_POINT,
            pipeline
        );
    }
}

macro_rules! dyn_fn {
    ($function:ident, $op_1:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*) => (
        pub fn $function(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, data_2, &mut pipeline);
            pipeline.finish();
            result
        }

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

dyn_fn!(
    bitwise_and_dyn,
    bitwise_and,
    bitwise_and_op_dyn,
    bitwise_and_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU,
    BooleanArrayGPU
);

dyn_fn!(
    bitwise_or_dyn,
    bitwise_or,
    bitwise_or_op_dyn,
    bitwise_or_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU,
    BooleanArrayGPU
);

dyn_fn!(
    bitwise_xor_dyn,
    bitwise_xor,
    bitwise_xor_op_dyn,
    bitwise_xor_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU,
    BooleanArrayGPU
);

macro_rules! dyn_fn_sh {
    ($function:ident, $op_1:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*) => (
        pub fn $function(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, data_2, &mut pipeline);
            pipeline.finish();
            result
        }

        pub fn $function_op(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (data_1, data_2) {
                $((ArrowArrayGPU::$y(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => arr_1.$op_2(arr_2, pipeline).into(),)+
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

dyn_fn_sh!(
    bitwise_shl_dyn,
    bitwise_shl,
    bitwise_shl_op_dyn,
    bitwise_shl_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU
);

dyn_fn_sh!(
    bitwise_shr_dyn,
    bitwise_shr,
    bitwise_shr_op_dyn,
    bitwise_shr_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU
);

macro_rules! dyn_not {
    ($function:ident, $op_1:ident, $function_op:ident, $op_2:ident, $( $y:ident ),*) => (
        pub fn $function(data_1: &ArrowArrayGPU) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(data_1.get_gpu_device(), None);
            let result = $function_op(data_1, &mut pipeline);
            pipeline.finish();
            result
        }

        pub fn $function_op(data_1: &ArrowArrayGPU, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (data_1) {
                $(ArrowArrayGPU::$y(arr_1) => arr_1.$op_2(pipeline).into(),)+
                _ => panic!(
                    "Operation {} not supported for type {:?}",
                    stringify!($function),
                    data_1.get_dtype(),
                ),
            }
        }
    )
}

dyn_not!(
    bitwise_not_dyn,
    bitwise_not,
    bitwise_not_op_dyn,
    bitwise_not_op,
    Int32ArrayGPU,
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    Int16ArrayGPU,
    UInt8ArrayGPU,
    Int8ArrayGPU,
    BooleanArrayGPU
);
