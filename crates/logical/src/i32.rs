use std::sync::Arc;

use arrow_gpu_array::array::{GpuDevice, Int32ArrayGPU, NullBitBufferGpu};

use crate::LogicalType;

const I32_LOGICAL_SHADER: &str = include_str!("../compute_shaders/i32/logical.wgsl");
const I32_NOT_SHADER: &str = include_str!("../compute_shaders/i32/not.wgsl");

impl LogicalType for i32 {
    type OutputType = Int32ArrayGPU;

    const SHADER: &'static str = I32_LOGICAL_SHADER;
    const NOT_SHADER: &'static str = I32_NOT_SHADER;

    fn create_new(
        data: Arc<wgpu::Buffer>,
        gpu_device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Int32ArrayGPU {
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
    use crate::Logical;

    use super::*;
    use arrow_gpu_test_macros::*;

    test_array_op!(
        test_bitwise_and_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        bitwise_and,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(-1), Some(100), Some(!100), None, Some(!450)],
        vec![Some(0), Some(1 & -1), Some(100), Some(0), None, None]
    );

    test_array_op!(
        test_bitwise_or_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        bitwise_or,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(-1), Some(100), Some(!100), None, Some(!450)],
        vec![
            Some(0),
            Some(1 | -1),
            Some(100),
            Some(100 | !100),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_xor_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        bitwise_xor,
        vec![Some(0), Some(1), Some(100), Some(100), Some(260), None],
        vec![Some(0), Some(-1), Some(100), Some(!100), None, Some(!450)],
        vec![
            Some(0),
            Some(1 ^ -1),
            Some(100 ^ 100),
            Some(100 ^ !100),
            None,
            None
        ]
    );

    test_unary_op!(
        test_bitwise_not_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        bitwise_not,
        vec![!0, !1, !2, !3, !4]
    );
}
