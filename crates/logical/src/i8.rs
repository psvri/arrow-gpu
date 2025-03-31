use crate::{
    LogicalType,
    i32::{I32_LOGICAL_SHADER, I32_NOT_SHADER},
};

const I8_SHIFT_SHADER: &str = include_str!("../compute_shaders/i8/shift.wgsl");

impl LogicalType for i8 {
    const SHADER: &'static str = I32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = I8_SHIFT_SHADER;
    const NOT_SHADER: &'static str = I32_NOT_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::UInt32ArrayGPU;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_i8_array_i8,
        Int8ArrayGPU,
        Int8ArrayGPU,
        Int8ArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(-50), None],
        [Some(0), Some(1), Some(-100), Some(!100), None, Some(!50)],
        [Some(0), Some(1), Some(100 & -100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_i8_array_i8,
        Int8ArrayGPU,
        Int8ArrayGPU,
        Int8ArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(-50), None],
        [Some(0), Some(1), Some(-100), Some(!100), None, Some(!50)],
        [
            Some(0),
            Some(1),
            Some(100 | -100),
            Some(100 | !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_xor_i8_array_i8,
        Int8ArrayGPU,
        Int8ArrayGPU,
        Int8ArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(-50), None],
        [Some(0), Some(0), Some(-100), Some(!100), None, Some(!50)],
        [
            Some(0),
            Some(1),
            Some(100 ^ -100),
            Some(100 ^ !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shl_i8_array_i8,
        Int8ArrayGPU,
        UInt32ArrayGPU,
        Int8ArrayGPU,
        bitwise_shl,
        bitwise_shl_dyn,
        [
            Some(0),
            Some(1),
            Some(100),
            Some(-100),
            Some(i8::MAX),
            Some(i8::MIN),
            Some(-50),
            None
        ],
        [
            Some(0),
            Some(1),
            Some(3),
            Some(3),
            Some(5),
            Some(5),
            None,
            Some(!50)
        ],
        [
            Some(0),
            Some(1 << 1),
            Some(100 << 3),
            Some(-100 << 3),
            Some(i8::MAX << 5),
            Some(i8::MIN << 5),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shr_i8_array_i8,
        Int8ArrayGPU,
        UInt32ArrayGPU,
        Int8ArrayGPU,
        bitwise_shr,
        bitwise_shr_dyn,
        [
            Some(0),
            Some(1),
            Some(100),
            Some(-100),
            Some(i8::MAX),
            Some(i8::MIN),
            Some(-50),
            None
        ],
        [
            Some(0),
            Some(1),
            Some(3),
            Some(3),
            Some(5),
            Some(5),
            None,
            Some(!50)
        ],
        [
            Some(0),
            Some(1 >> 1),
            Some(100 >> 3),
            Some(-100 >> 3),
            Some(i8::MAX >> 5),
            Some(i8::MIN >> 5),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_i8,
        Int8ArrayGPU,
        Int8ArrayGPU,
        [0, 1, 2, 3, 4, -1, -50],
        bitwise_not,
        bitwise_not_dyn,
        [!0, !1, !2, !3, !4, !-1, !-50]
    );
}
