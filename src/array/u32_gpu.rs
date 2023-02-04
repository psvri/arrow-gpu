use crate::kernels::arithmetic::*;
use async_trait::async_trait;
use std::sync::Arc;

use super::{gpu_ops::u32_ops::*, primitive_array_gpu::*, GpuDevice, NullBitBufferGpu};

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

impl_add_trait!(u32, add_scalar);
impl_sub_trait!(u32, sub_scalar);
impl_mul_trait!(u32, mul_scalar);
impl_div_trait!(u32, div_scalar);

impl_array_add_trait!(UInt32ArrayGPU, UInt32ArrayGPU, add_array_u32);

impl UInt32ArrayGPU {
    pub async fn braodcast(value: u32, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let data = Arc::new(braodcast_u32(&gpu_device, value, len.try_into().unwrap()).await);
        let null_buffer = None;

        Self {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_add_array!(
        test_add_u32_array_u32,
        UInt32ArrayGPU,
        vec![Some(0u32), Some(1), None, None, Some(4)],
        vec![Some(1u32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );

    test_scalar_op!(
        test_add_u32_scalar_u32,
        u32,
        vec![0, 1, 2, 3, 4],
        add,
        &100,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_u32_scalar_u32,
        u32,
        vec![0, 100, 200, 3, 104],
        sub,
        &100,
        vec![u32::MAX - 99, 0, 100, u32::MAX - 96, 4]
    );

    test_scalar_op!(
        test_mul_u32_scalar_u32,
        u32,
        vec![0, u32::MAX, 2, 3, 4],
        mul,
        &100,
        vec![0, u32::MAX - 99, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_u32_scalar_u32,
        u32,
        vec![0, 1, 100, 260, 450],
        div,
        &100,
        vec![0, 0, 1, 2, 4]
    );

    test_scalar_op!(
        test_div_by_zero_u32_scalar_u32,
        u32,
        vec![0, 1, 100, 260, 450],
        div,
        &0,
        vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
    );

    test_broadcast!(test_braodcast_u32, u32, 1);
}
