use std::sync::Arc;

use arrow_gpu_array::array::{GpuDevice, NullBitBufferGpu, UInt32ArrayGPU};

use crate::LogicalType;

const U32_LOGICAL_SHADER: &str = include_str!("../compute_shaders/u32/logical.wgsl");
const U32_NOT_SHADER: &str = include_str!("../compute_shaders/u32/not.wgsl");

impl LogicalType for u32 {
    type OutputType = UInt32ArrayGPU;

    const SHADER: &'static str = U32_LOGICAL_SHADER;
    const NOT_SHADER: &'static str = U32_NOT_SHADER;

    fn create_new(
        data: Arc<wgpu::Buffer>,
        gpu_device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        UInt32ArrayGPU {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{bitwise_xor_dyn, Logical};

    use super::*;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_and,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(1), Some(100), Some(!100), None, Some(!450)],
        vec![Some(0), Some(1), Some(100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_or,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(1), Some(100), Some(!100), None, Some(!450)],
        vec![Some(0), Some(1), Some(100), Some(100 | !100), None, None]
    );

    test_array_op!(
        test_bitwise_xor_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(0), Some(100), Some(!100), None, Some(!450)],
        vec![
            Some(0),
            Some(1),
            Some(100 ^ 100),
            Some(100 ^ !100),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        bitwise_not,
        vec![!0, !1, !2, !3, !4]
    );
}
