use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu, PrimitiveArrayGpu, UInt32ArrayGPU,
};
use merge::merge_null_buffers;
use put::apply_put_function;
use take::apply_take_function;

pub(crate) mod f32;
pub(crate) mod i16;
pub(crate) mod i32;
pub(crate) mod i8;
pub(crate) mod merge;
pub(crate) mod put;
pub(crate) mod take;
pub(crate) mod u16;
pub(crate) mod u32;
pub(crate) mod u8;

pub use merge::merge_dyn;
pub use put::put_dyn;
pub use take::take_dyn;

pub trait Swizzle {
    // Selects self incase of true else selects from other.
    // None values in mask results in None
    fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self;

    /// Take values from array by index.
    /// None values in mask results in None
    fn take(&self, indexes: &UInt32ArrayGPU) -> Self;

    /// Put values from src array to dst array.
    /// None values in mask results in None
    fn put(&self, src_indexes: &UInt32ArrayGPU, dst: &mut Self, dst_indexes: &UInt32ArrayGPU);
}

pub trait SwizzleType {
    const MERGE_SHADER: &'static str;
    const TAKE_SHADER: &'static str = todo!();
    const PUT_SHADER: &'static str = todo!();
}

impl<T: SwizzleType + ArrowPrimitiveType> Swizzle for PrimitiveArrayGpu<T> {
    fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self {
        let new_buffer = self.gpu_device.apply_ternary_function(
            &self.data,
            &other.data,
            &mask.data,
            T::ITEM_SIZE,
            T::MERGE_SHADER,
            "merge_array",
        );

        let op1 = self.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let op2 = other.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let mask_null = mask.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());

        let bit_buffer = merge_null_buffers(&self.gpu_device, op1, op2, &mask.data, mask_null);

        let new_null_buffer = bit_buffer.map(|buffer| NullBitBufferGpu {
            bit_buffer: Arc::new(buffer),
            len: self.len,
            gpu_device: self.gpu_device.clone(),
        });

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    fn take(&self, indexes: &UInt32ArrayGPU) -> Self {
        let new_buffer = apply_take_function(
            &self.gpu_device,
            &self.data,
            &indexes.data,
            indexes.len as u64,
            T::ITEM_SIZE,
            T::TAKE_SHADER,
            "take",
        );

        let new_null_buffer = match &self.null_buffer {
            Some(_) => todo!(),
            None => None,
        };

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: indexes.len,
            null_buffer: new_null_buffer,
        }
    }

    fn put(&self, src_indexes: &UInt32ArrayGPU, dst: &mut Self, dst_indexes: &UInt32ArrayGPU) {
        apply_put_function(
            &self.gpu_device,
            &self.data,
            &dst.data,
            &src_indexes.data,
            &dst_indexes.data,
            src_indexes.len as u64,
            T::PUT_SHADER,
            "put",
        );

        match (&self.null_buffer, &dst.null_buffer) {
            (None, None) => {}
            (None, Some(_)) => todo!(),
            (Some(_), None) => todo!(),
            (Some(_), Some(_)) => todo!(),
        }
    }
}
