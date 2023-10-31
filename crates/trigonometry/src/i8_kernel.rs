use std::sync::Arc;

use arrow_gpu_array::array::*;
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
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::*;

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
        vec![0, 1, -1, 56, -56, i8::MIN, i8::MAX],
        cos,
        cos_dyn,
        vec![
            0.0f32.cos(),
            1.0f32.cos(),
            -1.0f32.cos(),
            56.0f32.cos(),
            -56.0f32.cos(),
            (i8::MIN as f32).cos(),
            (i8::MAX as f32).cos(),
        ]
    );

    test_unary_op_float!(
        test_i8_sin,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, -1, 56, -56, i8::MIN, i8::MAX],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            -1.0f32.sin(),
            56.0f32.sin(),
            -56.0f32.sin(),
            (i8::MIN as f32).sin(),
            (i8::MAX as f32).sin(),
        ]
    );
}
