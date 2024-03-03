use std::sync::Arc;

use arrow_gpu_array::{
    array::{Int32ArrayGPU, NullBitBufferGpu},
    gpu_utils::GpuDevice,
};
use wgpu::Buffer;

use crate::MathUnaryType;

const I32UNARY_SHADER: &str = include_str!("../compute_shaders/i32/unary.wgsl");

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

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_test_macros::test_unary_op;
    use arrow_gpu_array::array::Int32ArrayGPU;

    test_unary_op!(
        test_i32_abs,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, -1, -2, 3, -4],
        abs,
        abs_dyn,
        vec![0, 1, 2, 3, 4]
    );
}
