use std::sync::Arc;

use arrow_gpu_array::array::{types::Int32Type, *};

use crate::impl_arithmetic_op;
use crate::*;

const I32_SCALAR_SHADER: &str = include_str!("../compute_shaders/i32/scalar.wgsl");
const I32_ARRAY_SHADER: &str = include_str!("../compute_shaders/i32/array.wgsl");

impl_arithmetic_op!(
    ArrowScalarAdd,
    Int32Type,
    add_scalar,
    Int32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_add"
);

impl_arithmetic_op!(
    ArrowScalarSub,
    Int32Type,
    sub_scalar,
    Int32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_sub"
);

impl_arithmetic_op!(
    ArrowScalarMul,
    Int32Type,
    mul_scalar,
    Int32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_mul"
);

impl_arithmetic_op!(
    ArrowScalarDiv,
    Int32Type,
    div_scalar,
    Int32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_div"
);

impl_arithmetic_op!(
    ArrowScalarRem,
    Int32Type,
    rem_scalar,
    Int32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_rem"
);

impl_arithmetic_op!(
    ArrowScalarAdd,
    Int32Type,
    add_scalar,
    Date32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_add"
);

impl_arithmetic_op!(
    ArrowScalarSub,
    Int32Type,
    sub_scalar,
    Date32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_sub"
);

impl_arithmetic_op!(
    ArrowScalarMul,
    Int32Type,
    mul_scalar,
    Date32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_mul"
);

impl_arithmetic_op!(
    ArrowScalarDiv,
    Int32Type,
    div_scalar,
    Date32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_div"
);

impl_arithmetic_op!(
    ArrowScalarRem,
    Int32Type,
    rem_scalar,
    Date32ArrayGPU,
    4,
    I32_SCALAR_SHADER,
    "i32_rem"
);

impl_arithmetic_array_op!(
    ArrowAdd,
    Int32Type,
    add,
    Int32ArrayGPU,
    4,
    I32_ARRAY_SHADER,
    "add_i32"
);

impl_arithmetic_array_op!(
    ArrowAdd,
    Int32Type,
    add,
    Date32ArrayGPU,
    4,
    I32_ARRAY_SHADER,
    "add_i32"
);

#[cfg(test)]
mod tests {
    use crate::rem_scalar_dyn;

    use super::*;
    use arrow_gpu_test_macros::{test_array_op, test_scalar_op};

    test_scalar_op!(
        test_add_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100i32,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 100, 200, 3, 104],
        sub_scalar,
        sub_scalar_dyn,
        100,
        vec![-100, 0, 100, -97, 4]
    );

    test_scalar_op!(
        test_mul_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, i32::MAX, 2, 3, 4],
        mul_scalar,
        mul_scalar_dyn,
        100,
        vec![0, -100, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        div_scalar_dyn,
        100,
        vec![0, 0, 1, 2, 4]
    );

    test_scalar_op!(
        test_rem_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 104],
        rem_scalar,
        rem_scalar_dyn,
        100i32,
        vec![0, 1, 2, 3, 4]
    );

    test_scalar_op!(
        test_rem_i32_scalar_date32,
        Int32ArrayGPU,
        Date32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 104],
        rem_scalar,
        rem_scalar_dyn,
        100i32,
        vec![0, 1, 2, 3, 4]
    );

    /*//ignore = "Not passing in linux CI but passes in windows 🤔"
    #[cfg(not(target_os = "linux"))]
    test_scalar_op!(
        test_div_by_zero_i32_scalar_i32,
        i32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        0,
        vec![-1; 5]
    );*/

    test_scalar_op!(
        test_rem_date32_scalar_i32,
        Date32ArrayGPU,
        Int32ArrayGPU,
        Date32ArrayGPU,
        vec![0, 1, 2, 3, 104],
        rem_scalar,
        rem_scalar_dyn,
        100i32,
        vec![0, 1, 2, 3, 4]
    );

    test_scalar_op!(
        test_rem_date32_scalar_date32,
        Date32ArrayGPU,
        Date32ArrayGPU,
        Date32ArrayGPU,
        vec![0, 1, 2, 3, 104],
        rem_scalar,
        rem_scalar_dyn,
        100i32,
        vec![0, 1, 2, 3, 4]
    );

    test_array_op!(
        test_add_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        add,
        vec![Some(0i32), Some(1), None, None, Some(4)],
        vec![Some(1i32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );

    test_array_op!(
        test_add_i32_array_date32,
        Int32ArrayGPU,
        Date32ArrayGPU,
        Date32ArrayGPU,
        add,
        vec![Some(0i32), Some(1), None, None, Some(4)],
        vec![Some(1i32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );
}
