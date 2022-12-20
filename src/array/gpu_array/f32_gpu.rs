use crate::kernels::add_ops::{ArrowAdd, ArrowAddAssign};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    gpu_ops::f32_ops::*,
    primitive_array_gpu::{impl_add_assign_trait, impl_add_trait, PrimitiveArrayGpu},
};

pub type Float32ArrayGPU = PrimitiveArrayGpu<f32>;

impl_add_trait!(f32, add_scalar);
impl_add_assign_trait!(f32, add_assign_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::gpu_array::primitive_array_gpu::test::{
        test_add_assign_scalar, test_add_scalar,
    };

    #[tokio::test]
    async fn test_large_f32_array() {
        let mut gpu_array = Float32ArrayGPU::from(
            &(0..1024 * 1024 * 10)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
        );
        let new_gpu_array = gpu_array.add(100.0).await;
        for (index, value) in new_gpu_array.raw_values().unwrap().into_iter().enumerate() {
            assert_eq!((index as f32) + 100.0, value);
        }
    }

    test_add_assign_scalar!(
        test_add_assign_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );

    test_add_assign_scalar!(
        test_add_assign_f32_option_scalar_f32,
        f32,
        vec![Some(0.0f32), Some(1.0), Some(2.0), None, None],
        100.0,
        vec![100.0, 101.0, 102.0, 100.0, 100.0],
        vec![Some(100.0), Some(101.0), Some(102.0), None, None]
    );

    test_add_scalar!(
        test_add_f32_scalar_f32,
        f32,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );
}
