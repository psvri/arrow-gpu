use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowPrimitiveType, BooleanArrayGPU, NullBitBufferGpu, PrimitiveArrayGpu, UInt32ArrayGPU,
};
use async_trait::async_trait;
use merge::merge_null_buffers;
use take::apply_take_function;

pub(crate) mod f32;
pub(crate) mod i16;
pub(crate) mod i32;
pub(crate) mod i8;
pub(crate) mod merge;
pub(crate) mod take;
pub(crate) mod u16;
pub(crate) mod u32;
pub(crate) mod u8;

pub use merge::merge_dyn;
pub use take::take_dyn;

#[async_trait]
pub trait Swizzle {
    // Selects self incase of true else selects from other.
    // None values in mask results in None
    async fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self;

    /// Take values from array by index.
    /// None values in mask results in None
    async fn take(&self, indexes: &UInt32ArrayGPU) -> Self;
}

pub trait SwizzleType {
    const MERGE_SHADER: &'static str;
    const TAKE_SHADER: &'static str = "";
}

#[async_trait]
impl<T: SwizzleType + ArrowPrimitiveType> Swizzle for PrimitiveArrayGpu<T> {
    async fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_ternary_function(
                &self.data,
                &other.data,
                &mask.data,
                T::ITEM_SIZE,
                T::MERGE_SHADER,
                "merge_array",
            )
            .await;

        let op1 = self.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let op2 = other.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let mask_null = mask.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());

        let bit_buffer =
            merge_null_buffers(&self.gpu_device, op1, op2, &mask.data, mask_null).await;

        let new_null_buffer = match bit_buffer {
            Some(buffer) => Some(NullBitBufferGpu {
                bit_buffer: Arc::new(buffer),
                len: self.len,
                gpu_device: self.gpu_device.clone(),
            }),
            None => None,
        };

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn take(&self, indexes: &UInt32ArrayGPU) -> Self {
        let new_buffer = apply_take_function(
            &self.gpu_device,
            &self.data,
            &indexes.data,
            indexes.len as u64,
            T::ITEM_SIZE,
            T::TAKE_SHADER,
            "take",
        )
        .await;

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
}

#[cfg(test)]
mod tests {
    use arrow_gpu_array::array;

    use once_cell::sync::Lazy;
    use pollster::FutureExt;
    use std::sync::Arc;

    use array::GpuDevice;

    pub static GPU_DEVICE: Lazy<Arc<GpuDevice>> =
        Lazy::new(|| Arc::new(GpuDevice::new().block_on()));
}