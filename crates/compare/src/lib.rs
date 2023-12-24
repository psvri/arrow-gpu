use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu, PrimitiveArrayGpu,
};

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

pub trait CompareType {
    const COMPARE_SHADER: &'static str;
    const MIN_MAX_SHADER: &'static str;
}

pub trait Compare {
    fn gt(&self, operand: &Self) -> BooleanArrayGPU;
    fn gteq(&self, operand: &Self) -> BooleanArrayGPU;
    fn lt(&self, operand: &Self) -> BooleanArrayGPU;
    fn lteq(&self, operand: &Self) -> BooleanArrayGPU;
    fn eq(&self, operand: &Self) -> BooleanArrayGPU;
}

pub trait MinMax {
    fn max(&self, operand: &Self) -> Self;
    fn min(&self, operand: &Self) -> Self;
}

macro_rules! apply_function {
    ($self: ident, $operand:ident, $entry_point: ident) => {
        let new_buffer = $self.gpu_device.apply_binary_function(
            &$self.data,
            &$operand.data,
            T::ITEM_SIZE,
            T::COMPARE_SHADER,
            $entry_point,
        );

        return BooleanArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            null_buffer: NullBitBufferGpu::merge_null_bit_buffer(
                &$self.null_buffer,
                &$operand.null_buffer,
            ),
        }
    };
}

macro_rules! apply_function_min_max {
    ($self: ident, $operand:ident, $entry_point: ident) => {
        let new_buffer = $self.gpu_device.apply_binary_function(
            &$self.data,
            &$operand.data,
            T::ITEM_SIZE,
            T::MIN_MAX_SHADER,
            $entry_point,
        );

        return Self {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            phantom: std::marker::PhantomData,
            null_buffer: NullBitBufferGpu::merge_null_bit_buffer(
                &$self.null_buffer,
                &$operand.null_buffer,
            ),
        }
    };
}

impl<T: CompareType + ArrowPrimitiveType> Compare for PrimitiveArrayGpu<T> {
    fn gt(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, GT_ENTRY_POINT);
    }

    fn gteq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, GTEQ_ENTRY_POINT);
    }

    fn lt(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, LT_ENTRY_POINT);
    }

    fn lteq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, LTEQ_ENTRY_POINT);
    }

    fn eq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, EQ_ENTRY_POINT);
    }
}

impl<T: CompareType + ArrowPrimitiveType> MinMax for PrimitiveArrayGpu<T> {
    fn min(&self, operand: &Self) -> Self {
        apply_function_min_max!(self, operand, MIN_ENTRY_POINT);
    }

    fn max(&self, operand: &Self) -> Self {
        apply_function_min_max!(self, operand, MAX_ENTRY_POINT);
    }
}

pub fn gt_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.gt(y),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.gt(y),
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn gteq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.gteq(y),
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.gteq(y),
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn lt_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.lt(y),
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.lt(y),
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn lteq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.lteq(y),
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.lteq(y),
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn eq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.eq(y),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.eq(y),
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn max_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.max(y).into(),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.max(y).into(),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.max(y).into(),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.max(y).into(),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.max(y).into(),
        _ => panic!(
            "Cannot compute max for types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub fn min_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.min(y).into(),
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.min(y).into(),
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.min(y).into(),
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.min(y).into(),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.min(y).into(),
        _ => panic!(
            "Cannot compute min for types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}
