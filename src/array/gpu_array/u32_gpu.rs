use crate::kernels::add_ops::ArrowAdd;
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    gpu_ops::u32_ops::*,
    primitive_array_gpu::{impl_add_assign_trait, impl_add_trait, PrimitiveArrayGpu},
    NullBitBufferGpu,
};
use crate::kernels::add_ops::ArrowAddAssign;

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

impl_add_trait!(u32, add_scalar);
impl_add_assign_trait!(u32, add_assign_scalar);

#[async_trait]
impl ArrowAdd<UInt32ArrayGPU> for UInt32ArrayGPU {
    type Output = Self;

    async fn add(&self, value: &UInt32ArrayGPU) -> Self::Output {
        assert!(Arc::ptr_eq(
            &self.gpu_device,
            &value.gpu_device
        ));
        let new_data_buffer = add_array(&self.gpu_device, &self.data, &value.data).await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &value.null_buffer).await;

        let result = Self {
            data: Arc::new(new_data_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: new_null_buffer,
        };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::gpu_array::primitive_array_gpu::test::*;

    test_add_assign_scalar!(
        test_add_assign_u32_scalar_u32,
        u32,
        vec![0u32, 1, 2, 3, 4],
        &100,
        vec![100, 101, 102, 103, 104]
    );

    test_add_assign_scalar!(
        test_add_assign_u32_option_scalar_u32,
        u32,
        vec![Some(0u32), Some(1), None, None, Some(4)],
        &100,
        vec![100, 101, 100, 100, 104],
        vec![Some(100u32), Some(101), None, None, Some(104)]
    );

    test_add_scalar!(
        test_add_u32_scalar_u32,
        u32,
        vec![0, 1, 2, 3, 4],
        &100,
        vec![100, 101, 102, 103, 104]
    );

    test_add_array!(
        test_add_u32_array_u32,
        UInt32ArrayGPU,
        vec![Some(0u32), Some(1), None, None, Some(4)],
        vec![Some(1u32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );
}
