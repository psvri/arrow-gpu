use crate::kernels::add_ops::ArrowAdd;
use async_trait::async_trait;
use std::sync::Arc;
use wgpu::{util::DeviceExt, Maintain};

use super::primitive_array_gpu::{add_assign_primitive, add_primitive, PrimitiveArrayGpu};
use crate::kernels::add_ops::ArrowAddAssign;

pub type Float32ArrayGPU = PrimitiveArrayGpu<f32>;
add_assign_primitive!(
    f32,
    "../../../compute_shaders/f32_assign_scalar.wgsl",
    1,
    "f32_add"
);

add_primitive!(
    f32,
    "../../../compute_shaders/f32_scalar.wgsl",
    1,
    "f32_add"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::gpu_array::primitive_array_gpu::test::{
        test_add_assign_scalar, test_add_scalar,
    };

    test_add_assign_scalar!(
        test_add_assign_f32_scalar_f32,
        f32,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );

    test_add_scalar!(
        test_add_f32_scalar_f32,
        f32,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );
}
