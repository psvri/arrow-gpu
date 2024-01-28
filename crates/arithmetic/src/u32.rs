use std::sync::Arc;

use arrow_gpu_array::array::{types::UInt32Type, *};

use crate::impl_arithmetic_op;
use crate::*;

const U32_SCALAR_SHADER: &str = include_str!("../compute_shaders/u32/scalar.wgsl");
const U32_ARRAY_SHADER: &str = include_str!("../compute_shaders/u32/array.wgsl");

impl_arithmetic_op!(
    ArrowScalarAdd,
    UInt32Type,
    add_scalar_op,
    UInt32ArrayGPU,
    U32_SCALAR_SHADER,
    "u32_add"
);

impl_arithmetic_op!(
    ArrowScalarSub,
    UInt32Type,
    sub_scalar_op,
    UInt32ArrayGPU,
    U32_SCALAR_SHADER,
    "u32_sub"
);

impl_arithmetic_op!(
    ArrowScalarMul,
    UInt32Type,
    mul_scalar_op,
    UInt32ArrayGPU,
    U32_SCALAR_SHADER,
    "u32_mul"
);

impl_arithmetic_op!(
    ArrowScalarDiv,
    UInt32Type,
    div_scalar_op,
    UInt32ArrayGPU,
    U32_SCALAR_SHADER,
    "u32_div"
);

impl_arithmetic_op!(
    ArrowScalarRem,
    UInt32Type,
    rem_scalar_op,
    UInt32ArrayGPU,
    U32_SCALAR_SHADER,
    "u32_rem"
);

impl_arithmetic_array_op!(
    ArrowAdd,
    UInt32Type,
    add_op,
    UInt32ArrayGPU,
    U32_ARRAY_SHADER,
    "add_u32"
);

#[cfg(test)]
mod test {
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::{test_array_op, test_scalar_op};

    use super::*;

    test_scalar_op!(
        test_add_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100u32,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 100, 200, 3, 104],
        sub_scalar,
        sub_scalar_dyn,
        100,
        vec![u32::MAX - 99, 0, 100, u32::MAX - 96, 4]
    );

    test_scalar_op!(
        test_mul_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, u32::MAX, 2, 3, 4],
        mul_scalar,
        mul_scalar_dyn,
        100,
        vec![0, u32::MAX - 99, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        div_scalar_dyn,
        100,
        vec![0, 0, 1, 2, 4]
    );

    test_scalar_op!(
        test_rem_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 100, 260, 450],
        rem_scalar,
        rem_scalar_dyn,
        100,
        vec![0, 1, 0, 60, 50]
    );

    /*test_scalar_op!(
        test_div_by_zero_u32_scalar_u32,
        u32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        0,
        vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
    );*/

    test_array_op!(
        test_add_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        add,
        vec![Some(0u32), Some(1), None, None, Some(4)],
        vec![Some(1u32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );
}
