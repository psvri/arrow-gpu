use crate::{
    i32::{I32_LOGICAL_SHADER, I32_NOT_SHADER},
    LogicalType,
};

const I16_SHIFT_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/shift.wgsl")
);

impl LogicalType for i16 {
    const SHADER: &'static str = I32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = I16_SHIFT_SHADER;
    const NOT_SHADER: &'static str = I32_NOT_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::UInt32ArrayGPU;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_i16_array_i16,
        Int16ArrayGPU,
        Int16ArrayGPU,
        Int16ArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(1), Some(-100), Some(!100), None, Some(!450)],
        vec![Some(0), Some(1), Some(100 & -100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_i16_array_i16,
        Int16ArrayGPU,
        Int16ArrayGPU,
        Int16ArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(1), Some(-100), Some(!100), None, Some(!450)],
        vec![
            Some(0),
            Some(1),
            Some(100 | -100),
            Some(100 | !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_xor_i16_array_i16,
        Int16ArrayGPU,
        Int16ArrayGPU,
        Int16ArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(0), Some(-100), Some(!100), None, Some(!450)],
        vec![
            Some(0),
            Some(1),
            Some(100 ^ -100),
            Some(100 ^ !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shl_i16_array_i16,
        Int16ArrayGPU,
        UInt32ArrayGPU,
        Int16ArrayGPU,
        bitwise_shl,
        bitwise_shl_dyn,
        vec![
            Some(0),
            Some(1),
            Some(-100),
            Some(i16::MAX),
            Some(260),
            None
        ],
        vec![Some(0), Some(1), Some(3), Some(5), None, Some(!450)],
        vec![
            Some(0),
            Some(1 << 1),
            Some(-100 << 3),
            Some(i16::MAX << 5),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shr_i16_array_i16,
        Int16ArrayGPU,
        UInt32ArrayGPU,
        Int16ArrayGPU,
        bitwise_shr,
        bitwise_shr_dyn,
        vec![
            Some(0),
            Some(1),
            Some(-100),
            Some(-100),
            Some(i16::MAX),
            Some(260),
            None
        ],
        vec![
            Some(0),
            Some(1),
            Some(3),
            Some(5),
            Some(5),
            None,
            Some(!450)
        ],
        vec![
            Some(0),
            Some(1 >> 1),
            Some(-100 >> 3),
            Some(-100 >> 5),
            Some(i16::MAX >> 5),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_i16,
        Int16ArrayGPU,
        Int16ArrayGPU,
        vec![0, 1, 2, -3, 4],
        bitwise_not,
        bitwise_not_dyn,
        vec![!0, !1, !2, !-3, !4]
    );
}
