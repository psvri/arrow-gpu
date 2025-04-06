use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;

use crate::Cast;
use crate::impl_cast;

const U8_CAST_U16_SHADER: &str = include_str!("../compute_shaders/u8/cast_u16.wgsl");
const U8_CAST_U32_SHADER: &str = include_str!("../compute_shaders/u8/cast_u32.wgsl");
const U8_CAST_F32_SHADER: &str = include_str!("../compute_shaders/u8/cast_f32.wgsl");

impl_cast!(Int8ArrayGPU, UInt8ArrayGPU);

impl_cast!(
    Int16ArrayGPU,
    UInt8ArrayGPU,
    U8_CAST_U16_SHADER,
    "cast_u16",
    1,
    2
);

impl_cast!(
    Int32ArrayGPU,
    UInt8ArrayGPU,
    U8_CAST_U32_SHADER,
    "cast_u32",
    1,
    4
);

impl_cast!(
    UInt16ArrayGPU,
    UInt8ArrayGPU,
    U8_CAST_U16_SHADER,
    "cast_u16",
    1,
    2
);

impl_cast!(
    UInt32ArrayGPU,
    UInt8ArrayGPU,
    U8_CAST_U32_SHADER,
    "cast_u32",
    1,
    4
);

impl_cast!(
    Float32ArrayGPU,
    UInt8ArrayGPU,
    U8_CAST_F32_SHADER,
    "cast_f32",
    1,
    4
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;

    test_cast_op!(
        test_cast_u8_to_i8,
        UInt8ArrayGPU,
        Int8ArrayGPU,
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
        ],
        Int8Type,
        [0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_u8_to_i16,
        UInt8ArrayGPU,
        Int16ArrayGPU,
        [0, 1, 2, 3, 255, 254, 253, 249, 7],
        Int16Type,
        [0, 1, 2, 3, 255, 254, 253, 249, 7]
    );

    test_cast_op!(
        test_cast_u8_to_i32,
        UInt8ArrayGPU,
        Int32ArrayGPU,
        [0, 1, 2, 3, 255, 254, 253, 249, 7],
        Int32Type,
        [0, 1, 2, 3, 255, 254, 253, 249, 7]
    );

    test_cast_op!(
        test_cast_u8_to_u16,
        UInt8ArrayGPU,
        UInt16ArrayGPU,
        [0, 1, 2, 3, 255, 250, 7],
        UInt16Type,
        [0, 1, 2, 3, 255, 250, 7]
    );

    test_cast_op!(
        test_cast_u8_to_u32,
        UInt8ArrayGPU,
        UInt32ArrayGPU,
        [0, 1, 2, 3, 255, 250, 7],
        UInt32Type,
        [0, 1, 2, 3, 255, 250, 7]
    );

    test_cast_op!(
        test_cast_u8_to_f32,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        [0, 1, 2, 3, 255, 250, 7],
        Float32Type,
        [0.0, 1.0, 2.0, 3.0, 255.0, 250.0, 7.0]
    );
}
