use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu, PrimitiveArrayGpu,
};
use async_trait::async_trait;

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

#[async_trait]
pub trait Compare {
    async fn gt(&self, operand: &Self) -> BooleanArrayGPU;
    async fn gteq(&self, operand: &Self) -> BooleanArrayGPU;
    async fn lt(&self, operand: &Self) -> BooleanArrayGPU;
    async fn lteq(&self, operand: &Self) -> BooleanArrayGPU;
    async fn eq(&self, operand: &Self) -> BooleanArrayGPU;
}

#[async_trait]
pub trait MinMax {
    async fn max(&self, operand: &Self) -> Self;
    async fn min(&self, operand: &Self) -> Self;
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

#[async_trait]
impl<T: CompareType + ArrowPrimitiveType> Compare for PrimitiveArrayGpu<T> {
    async fn gt(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, GT_ENTRY_POINT);
    }

    async fn gteq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, GTEQ_ENTRY_POINT);
    }

    async fn lt(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, LT_ENTRY_POINT);
    }

    async fn lteq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, LTEQ_ENTRY_POINT);
    }

    async fn eq(&self, operand: &Self) -> BooleanArrayGPU {
        apply_function!(self, operand, EQ_ENTRY_POINT);
    }
}

#[async_trait]
impl<T: CompareType + ArrowPrimitiveType> MinMax for PrimitiveArrayGpu<T> {
    async fn min(&self, operand: &Self) -> Self {
        apply_function_min_max!(self, operand, MIN_ENTRY_POINT);
    }

    async fn max(&self, operand: &Self) -> Self {
        apply_function_min_max!(self, operand, MAX_ENTRY_POINT);
    }
}

pub async fn gt_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.gt(y).await,
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.gt(y).await,
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn gteq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.gteq(y).await,
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.gteq(y).await,
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn lt_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.lt(y).await,
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.lt(y).await,
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn lteq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.lteq(y).await,
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.lteq(y).await,
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn eq_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> BooleanArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowArrayGPU::UInt8ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowArrayGPU::Int16ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowArrayGPU::Int8ArrayGPU(y)) => x.eq(y).await,
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => x.eq(y).await,
        _ => panic!(
            "Cannot compare types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn max_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => {
            x.max(y).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => {
            x.max(y).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => {
            x.max(y).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.max(y).await.into(),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => {
            x.max(y).await.into()
        }
        _ => panic!(
            "Cannot compute max for types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}

pub async fn min_dyn(x: &ArrowArrayGPU, y: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (x, y) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => {
            x.min(y).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => {
            x.min(y).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowArrayGPU::UInt16ArrayGPU(y)) => {
            x.min(y).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => x.min(y).await.into(),
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => {
            x.min(y).await.into()
        }
        _ => panic!(
            "Cannot compute min for types {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype()
        ),
    }
}
