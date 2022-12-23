use crate::kernels::arithmetic::*;
use async_trait::async_trait;
use std::sync::Arc;

use super::{gpu_ops::f32_ops::*, primitive_array_gpu::*, NullBitBufferGpu};

pub type Float32ArrayGPU = PrimitiveArrayGpu<f32>;

impl_add_trait!(f32, add_scalar);
impl_sub_trait!(f32, sub_scalar);
impl_mul_trait!(f32, mul_scalar);
impl_div_trait!(f32, div_scalar);

impl_array_add_trait!(Float32ArrayGPU, Float32ArrayGPU, add_array_f32);

impl_add_assign_trait!(f32, add_assign_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::gpu_array::primitive_array_gpu::test::*;

    #[tokio::test]
    async fn test_large_f32_array() {
        let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
        let gpu_array = Float32ArrayGPU::from_vec(
            &(0..1024 * 1024 * 10)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            device,
        );
        let new_gpu_array = gpu_array.add(&100.0).await;
        for (index, value) in new_gpu_array.raw_values().unwrap().into_iter().enumerate() {
            assert_eq!((index as f32) + 100.0, value);
        }
    }

    #[tokio::test]
    async fn test_f32_array_from_optinal_vec() {
        let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
        let gpu_array_1 = Float32ArrayGPU::from_optional_vec(
            &vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
            device.clone(),
        );
        assert_eq!(
            gpu_array_1.raw_values().unwrap(),
            vec![0.0, 1.0, 0.0, 0.0, 4.0]
        );
        assert_eq!(
            gpu_array_1.null_buffer.raw_values().unwrap(),
            vec![0b00010011]
        );
        let gpu_array_2 = Float32ArrayGPU::from_optional_vec(
            &vec![Some(1.0), Some(2.0), None, Some(4.0), None],
            device,
        );
        assert_eq!(
            gpu_array_2.raw_values().unwrap(),
            vec![1.0, 2.0, 0.0, 4.0, 0.0]
        );
        assert_eq!(
            gpu_array_2.null_buffer.raw_values().unwrap(),
            vec![0b00001011]
        );
        let new_bit_buffer = NullBitBufferGpu::merge_null_bit_buffer(
            &gpu_array_2.null_buffer,
            &gpu_array_1.null_buffer,
        )
        .await;
        assert_eq!(new_bit_buffer.raw_values().unwrap(), vec![0b00000011]);
    }

    test_add_assign_scalar!(
        test_add_assign_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        &100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );

    test_add_assign_scalar!(
        test_add_assign_f32_option_scalar_f32,
        f32,
        vec![Some(0.0f32), Some(1.0), Some(2.0), None, None],
        &100.0,
        vec![100.0, 101.0, 102.0, 100.0, 100.0],
        vec![Some(100.0), Some(101.0), Some(102.0), None, None]
    );

    test_add_array!(
        test_add_u32_array_u32,
        Float32ArrayGPU,
        vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
        vec![Some(1.0), Some(2.0), None, Some(4.0), None],
        vec![Some(1.0), Some(3.0), None, None, None]
    );

    test_scalar_op!(
        test_add_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        add,
        &100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );

    test_scalar_op!(
        test_div_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        div,
        &100.0,
        vec![0.0, 0.01, 0.02, 0.03, 0.04]
    );

    test_scalar_op!(
        test_mul_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        mul,
        &100.0,
        vec![0.0, 100.0, 200.0, 300.0, 400.0]
    );

    test_scalar_op!(
        test_sub_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        sub,
        &100.0,
        vec![-100.0, -99.0, -98.0, -97.0, -96.0]
    );
}
