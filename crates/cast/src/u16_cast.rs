use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;

use crate::impl_cast;
use crate::Cast;

const U16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cast_u32.wgsl")
);
const U16_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cast_f32.wgsl")
);

impl_cast!(Int16ArrayGPU, UInt16ArrayGPU);

impl_cast!(
    Int32ArrayGPU,
    UInt16ArrayGPU,
    U16_CAST_I32_SHADER,
    "cast_u32",
    2,
    2
);

impl_cast!(
    UInt32ArrayGPU,
    UInt16ArrayGPU,
    U16_CAST_I32_SHADER,
    "cast_u32",
    2,
    2
);

impl_cast!(
    Float32ArrayGPU,
    UInt16ArrayGPU,
    U16_CAST_F32_SHADER,
    "cast_f32",
    2,
    2
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;

    test_cast_op!(
        test_cast_u16_to_i16,
        UInt16ArrayGPU,
        Int16ArrayGPU,
        [0, 1, 2, 3, 4, u16::MAX, u16::MAX - 1, u16::MAX - 2],
        Int16Type,
        [0, 1, 2, 3, 4, -1, -2, -3]
    );

    test_cast_op!(
        test_cast_u16_to_i32,
        UInt16ArrayGPU,
        Int32ArrayGPU,
        [0, 1, 2, 3, 4, u16::MAX, u16::MAX - 1, u16::MAX - 2],
        Int32Type,
        [
            0,
            1,
            2,
            3,
            4,
            u16::MAX as i32,
            (u16::MAX - 1) as i32,
            (u16::MAX - 2) as i32
        ]
    );

    test_cast_op!(
        test_cast_u16_to_u32,
        UInt16ArrayGPU,
        UInt32ArrayGPU,
        [0, 1, 2, 3, 4],
        UInt32Type,
        [0, 1, 2, 3, 4]
    );

    test_cast_op!(
        test_cast_u16_to_f32,
        UInt16ArrayGPU,
        Float32ArrayGPU,
        [0, 1, 2, 3, 4, u16::MAX],
        Float32Type,
        [0.0, 1.0, 2.0, 3.0, 4.0, u16::MAX as f32]
    );
}
