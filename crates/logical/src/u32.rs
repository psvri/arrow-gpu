use crate::LogicalType;

pub(crate) const U32_LOGICAL_SHADER: &str = include_str!("../compute_shaders/u32/logical.wgsl");
pub(crate) const U32_NOT_SHADER: &str = include_str!("../compute_shaders/u32/not.wgsl");
pub(crate) const U32_SHIFT_SHADER: &str = include_str!("../compute_shaders/u32/shift.wgsl");

impl LogicalType for u32 {
    const SHADER: &'static str = U32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = U32_SHIFT_SHADER;
    const NOT_SHADER: &'static str = U32_NOT_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(260), None],
        [Some(0), Some(1), Some(100), Some(!100), None, Some(!450)],
        [Some(0), Some(1), Some(100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(260), None],
        [Some(0), Some(1), Some(100), Some(!100), None, Some(!450)],
        [Some(0), Some(1), Some(100), Some(100 | !100), None, None]
    );

    test_array_op!(
        test_bitwise_xor_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        [Some(0), Some(1), Some(100), Some(100), Some(260), None],
        [Some(0), Some(0), Some(100), Some(!100), None, Some(!450)],
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
        test_bitwise_shl_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_shl,
        bitwise_shl_dyn,
        [Some(0), Some(1), Some(100), Some(u32::MAX), Some(260), None],
        [Some(0), Some(1), Some(3), Some(5), None, Some(!450)],
        [
            Some(0),
            Some(1 << 1),
            Some(100 << 3),
            Some(u32::MAX << 5),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_shr_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_shr,
        bitwise_shr_dyn,
        [Some(0), Some(1), Some(100), Some(u32::MAX), Some(260), None],
        [Some(0), Some(1), Some(3), Some(5), None, Some(!450)],
        [
            Some(0),
            Some(1 >> 1),
            Some(100 >> 3),
            Some(u32::MAX >> 5),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        [0, 1, 2, 3, 4],
        bitwise_not,
        bitwise_not_dyn,
        [!0, !1, !2, !3, !4]
    );
}
