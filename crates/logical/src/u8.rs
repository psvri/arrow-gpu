use crate::{
    LogicalType,
    u32::{U32_LOGICAL_SHADER, U32_NOT_SHADER},
};

const U8_SHIFT_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/shift.wgsl")
);

impl LogicalType for u8 {
    const SHADER: &'static str = U32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = U8_SHIFT_SHADER;
    const NOT_SHADER: &'static str = U32_NOT_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::UInt32ArrayGPU;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(150), None],
        [Some(0), Some(1), Some(100), Some(!100), None, Some(!50)],
        [Some(0), Some(1), Some(100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(150), None],
        [Some(0), Some(1), Some(100), Some(!100), None, Some(!50)],
        [Some(0), Some(1), Some(100), Some(100 | !100), None, None]
    );

    test_array_op!(
        test_bitwise_xor_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(150), None],
        [Some(0), Some(0), Some(100), Some(!100), None, Some(!50)],
        [
            Some(0),
            Some(1),
            Some(100 ^ 100),
            Some(100 ^ !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shl_u8_array_u8,
        UInt8ArrayGPU,
        UInt32ArrayGPU,
        UInt8ArrayGPU,
        bitwise_shl,
        bitwise_shl_dyn,
        [Some(0), Some(1), Some(100), Some(u8::MAX), Some(150), None],
        [Some(0), Some(1), Some(3), Some(5), None, Some(!50)],
        [
            Some(0),
            Some(1 << 1),
            Some(100 << 3),
            Some(u8::MAX << 5),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shr_u8_array_u8,
        UInt8ArrayGPU,
        UInt32ArrayGPU,
        UInt8ArrayGPU,
        bitwise_shr,
        bitwise_shr_dyn,
        [Some(0), Some(1), Some(100), Some(u8::MAX), Some(150), None],
        [Some(0), Some(1), Some(3), Some(5), None, Some(!50)],
        [
            Some(0),
            Some(1 >> 1),
            Some(100 >> 3),
            Some(u8::MAX >> 5),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        [0, 1, 2, 3, 4],
        bitwise_not,
        bitwise_not_dyn,
        [!0, !1, !2, !3, !4]
    );
}
