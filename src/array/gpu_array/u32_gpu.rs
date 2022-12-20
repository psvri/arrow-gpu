use crate::kernels::add_ops::ArrowAdd;
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    gpu_ops::u32_ops::*,
    primitive_array_gpu::{impl_add_assign_trait, impl_add_trait, PrimitiveArrayGpu},
};
use crate::kernels::add_ops::ArrowAddAssign;

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

impl_add_trait!(u32, add_scalar);
impl_add_assign_trait!(u32, add_assign_scalar);

/*add_assign_primitive!(
    u32,
    "../../../compute_shaders/u32_assign_scalar.wgsl",
    1,
    "u32_add_assign"
);*/

/*add_primitive!(
    u32,
    "../../../compute_shaders/u32_scalar.wgsl",
    1,
    "u32_add"
);*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::gpu_array::primitive_array_gpu::test::{
        test_add_assign_scalar, test_add_scalar,
    };

    test_add_assign_scalar!(
        test_add_assign_u32_scalar_u32,
        u32,
        vec![0, 1, 2, 3, 4],
        100,
        vec![100, 101, 102, 103, 104]
    );

    test_add_assign_scalar!(
        test_add_assign_u32_option_scalar_u32,
        u32,
        vec![Some(0), Some(1), None, None, Some(4)],
        100,
        vec![100, 101, 100, 100, 104]
    );

    test_add_scalar!(
        test_add_u32_scalar_u32,
        u32,
        vec![0, 1, 2, 3, 4],
        100,
        vec![100, 101, 102, 103, 104]
    );
}
