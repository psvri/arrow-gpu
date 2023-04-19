use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu, PrimitiveArrayGpu,
};
use async_trait::async_trait;

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

pub trait CompareType {
    const COMPARE_SHADER: &'static str;
}

#[async_trait]
pub trait Compare {
    async fn gt(&self, operand: &Self) -> BooleanArrayGPU;
    async fn gteq(&self, operand: &Self) -> BooleanArrayGPU;
    async fn lt(&self, operand: &Self) -> BooleanArrayGPU;
    async fn lteq(&self, operand: &Self) -> BooleanArrayGPU;
    async fn eq(&self, operand: &Self) -> BooleanArrayGPU;
}

macro_rules! apply_function {
    ($self: ident, $operand:ident, $entry_point: ident) => {
        let new_buffer = $self
            .gpu_device
            .apply_binary_function(
                &$self.data,
                &$operand.data,
                T::ITEM_SIZE,
                T::COMPARE_SHADER,
                $entry_point,
            )
            .await;

        return BooleanArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            null_buffer: NullBitBufferGpu::merge_null_bit_buffer(
                &$self.null_buffer,
                &$operand.null_buffer,
            )
            .await,
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
