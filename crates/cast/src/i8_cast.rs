use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;

use crate::Cast;
use crate::impl_cast;

const I8_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_i32.wgsl")
);
const I8_CAST_I16_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_i16.wgsl")
);
const I8_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_f32.wgsl")
);

impl_cast!(
    Int32ArrayGPU,
    Int8ArrayGPU,
    I8_CAST_I32_SHADER,
    "cast_i32",
    1,
    4
);

impl_cast!(
    UInt32ArrayGPU,
    Int8ArrayGPU,
    I8_CAST_I32_SHADER,
    "cast_i32",
    1,
    4
);

impl_cast!(
    Int16ArrayGPU,
    Int8ArrayGPU,
    I8_CAST_I16_SHADER,
    "cast_i16",
    1,
    2
);

impl_cast!(
    UInt16ArrayGPU,
    Int8ArrayGPU,
    I8_CAST_I16_SHADER,
    "cast_i16",
    1,
    2
);

impl_cast!(
    Float32ArrayGPU,
    Int8ArrayGPU,
    I8_CAST_F32_SHADER,
    "cast_f32",
    1,
    4
);

impl_cast!(UInt8ArrayGPU, Int8ArrayGPU);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;

    test_cast_op!(
        test_cast_i8_to_i32,
        Int8ArrayGPU,
        Int32ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        Int32Type,
        [0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_i8_to_u32,
        Int8ArrayGPU,
        UInt32ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt32Type,
        [
            0,
            1,
            2,
            3,
            u32::MAX,
            u32::MAX - 1,
            u32::MAX - 2,
            u32::MAX - 6,
            7
        ]
    );

    test_cast_op!(
        test_cast_i8_to_i16,
        Int8ArrayGPU,
        Int16ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        Int16Type,
        [0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_i8_to_u16,
        Int8ArrayGPU,
        UInt16ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt16Type,
        [
            0,
            1,
            2,
            3,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 2,
            u16::MAX - 6,
            7
        ]
    );

    test_cast_op!(
        test_cast_i8_to_u8,
        Int8ArrayGPU,
        UInt8ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt8Type,
        [
            0,
            1,
            2,
            3,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 2,
            u8::MAX - 6,
            7
        ]
    );

    test_cast_op!(
        test_cast_i8_to_f32,
        Int8ArrayGPU,
        Float32ArrayGPU,
        [0, 1, 2, 3, -1, -2, -3, -7, 7],
        Float32Type,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -7.0, 7.0]
    );
}
