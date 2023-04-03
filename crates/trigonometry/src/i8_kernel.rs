use std::sync::Arc;

use arrow_gpu_array::array::{
    f32_gpu::Float32ArrayGPU, gpu_device::GpuDevice, null_bit_buffer::NullBitBufferGpu,
};
use wgpu::Buffer;

use crate::{HyperbolicType, TrigonometricType};

const HYPERBOLIC_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/hyperbolic.wgsl")
);

const TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/trigonometry.wgsl")
);

impl HyperbolicType for i8 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = HYPERBOLIC_SHADER;
    const TYPE_STR: &'static str = "i8";
    const BUFFER_SIZE_MULTIPLIER: u64 = 4;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Float32ArrayGPU {
            data,
            gpu_device: device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

impl TrigonometricType for i8 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = TRIGONOMETRY_SHADER;
    const TYPE_STR: &'static str = "i8";
    const BUFFER_SIZE_MULTIPLIER: u64 = 4;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Float32ArrayGPU {
            data,
            gpu_device: device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_gpu_array::array::i8_gpu::Int8ArrayGPU;
    use arrow_gpu_array::array::{f32_gpu::Float32ArrayGPU, gpu_device::GpuDevice};
    use arrow_gpu_test_macros::*;
    use std::sync::Arc;

    use crate::*;

    test_unary_op_float!(
        test_i8_sinh,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3],
        sinh,
        sinh_dyn,
        vec![
            0.0f32.sinh(),
            1.0f32.sinh(),
            2.0f32.sinh(),
            3.0f32.sinh(),
            -1.0f32.sinh(),
            -2.0f32.sinh(),
            -3.0f32.sinh()
        ]
    );

    //TODO: Fix negative values test failure
    test_unary_op_float!(
        test_i8_cos,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3],
        cos,
        cos_dyn,
        vec![0.0f32.cos(), 1.0f32.cos(), 2.0f32.cos(), 3.0f32.cos()]
    );

    test_unary_op_float!(
        test_i8_sin,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 5, -7, -8],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            5.0f32.sin(),
            -7.0f32.sin(),
            -8.0f32.sin(),
        ]
    );
}