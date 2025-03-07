use std::sync::Arc;

use arrow_gpu_array::array::{ArrayUtils, Float32ArrayGPU, NullBitBufferGpu, UInt32ArrayGPU};
use arrow_gpu_array::gpu_utils::*;

use crate::{BitCast, impl_bitcast};

impl_bitcast!(Float32ArrayGPU, UInt32ArrayGPU);

#[cfg(test)]
mod test {
    use crate::ArrowType;
    use crate::BitCast;
    use crate::bitcast_dyn;
    use crate::tests::test_bitcast_op;
    use arrow_gpu_array::array::{Float32ArrayGPU, UInt32ArrayGPU};

    test_bitcast_op!(
        test_bitcast_u32_to_f32,
        UInt32ArrayGPU,
        Float32ArrayGPU,
        [0, 1, 10, 5713, 57130, u32::MIN, u32::MAX],
        Float32Type,
        [
            0.0,
            f32::from_bits(1),
            f32::from_bits(10),
            f32::from_bits(5713),
            f32::from_bits(57130),
            f32::from_bits(u32::MIN),
            f32::from_bits(u32::MAX)
        ]
    );
}
