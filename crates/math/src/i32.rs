use std::sync::Arc;

use arrow_gpu_array::{
    array::{Int32ArrayGPU, NullBitBufferGpu},
    gpu_utils::GpuDevice,
};
use wgpu::Buffer;

use crate::{MathBinaryType, MathUnaryType};

const I32UNARY_SHADER: &str = include_str!("../compute_shaders/i32/unary.wgsl");
const I32BINARY_SHADER: &str = include_str!("../compute_shaders/i32/binary.wgsl");

impl MathUnaryType for i32 {
    type OutputType = Int32ArrayGPU;

    const SHADER: &'static str = I32UNARY_SHADER;
    const BUFFER_SIZE_MULTIPLIER: u64 = 1;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Int32ArrayGPU {
            data,
            gpu_device: device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

impl MathBinaryType for i32 {
    type OutputType = Int32ArrayGPU;

    const SHADER: &'static str = I32BINARY_SHADER;
    const BUFFER_SIZE_MULTIPLIER: u64 = 1;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Int32ArrayGPU {
            data,
            gpu_device: device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::Int32ArrayGPU;
    use arrow_gpu_test_macros::*;

    test_unary_op!(
        test_i32_abs,
        Int32ArrayGPU,
        Int32ArrayGPU,
        [0, -1, -2, 3, -4],
        abs,
        abs_dyn,
        [0, 1, 2, 3, 4]
    );

    test_array_op!(
        test_i32_power,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        power,
        power_dyn,
        [
            Some(0i32),
            Some(-1),
            Some(-2),
            Some(-3),
            Some(-4),
            Some(-2),
            None,
            Some(1),
            None
        ],
        [
            Some(0i32),
            Some(-1),
            Some(2),
            Some(3),
            Some(1),
            Some(-2),
            Some(1),
            None,
            None
        ],
        [
            Some(1i32),
            Some(-1),
            Some(4),
            Some(-27),
            Some(-4),
            Some(0),
            None,
            None,
            None
        ]
    );
}
