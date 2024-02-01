use arrow_gpu_array::array::{types::*, *};
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;

use crate::*;

const U16_SCALAR_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/scalar.wgsl")
);

impl_arithmetic_op!(
    ArrowScalarAdd,
    UInt16Type,
    add_scalar_op,
    UInt16ArrayGPU,
    U16_SCALAR_SHADER,
    "u16_add"
);

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu_test_macros::test_scalar_op;

    test_scalar_op!(
        test_add_u16_scalar_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100u16,
        vec![100, 101, 102, 103, 104]
    );
}
