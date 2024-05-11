use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;
use wgpu::Buffer;

use crate::{HyperbolicType, TrigonometricType};

const HYPERBOLIC_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/hyperbolic.wgsl")
);

const TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/trigonometry.wgsl")
);

impl HyperbolicType for i16 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = HYPERBOLIC_SHADER;
    const TYPE_STR: &'static str = "i16";
    const BUFFER_SIZE_MULTIPLIER: u64 = 2;

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

impl TrigonometricType for i16 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = TRIGONOMETRY_SHADER;
    const TYPE_STR: &'static str = "i16";
    const BUFFER_SIZE_MULTIPLIER: u64 = 2;

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
        test_i16_sinh,
        Int16ArrayGPU,
        Float32ArrayGPU,
        [0, 1, 2, 3, -1, -2, 4, -5, 7],
        sinh,
        sinh_dyn,
        [
            0.0f32.sinh(),
            1.0f32.sinh(),
            2.0f32.sinh(),
            3.0f32.sinh(),
            (-1.0f32).sinh(),
            (-2.0f32).sinh(),
            4.0f32.sinh(),
            (-5.0f32).sinh(),
            7.0f32.sinh()
        ]
    );

    //TODO: Fix negative values test failure
    test_unary_op_float!(
        test_i16_cos,
        Int16ArrayGPU,
        Float32ArrayGPU,
        [0, 1, -1, 4096, -4096, i16::MIN, i16::MAX],
        cos,
        cos_dyn,
        [
            0f32.cos(),
            1f32.cos(),
            (-1f32).cos(),
            4096f32.cos(),
            (-4096f32).cos(),
            (i16::MIN as f32).cos(),
            (i16::MAX as f32).cos()
        ]
    );

    test_unary_op_float!(
        test_i16_sin,
        Int16ArrayGPU,
        Float32ArrayGPU,
        [0, 1, -1, 4096, -4096, i16::MIN, i16::MAX],
        sin,
        sin_dyn,
        [
            0f32.sin(),
            1f32.sin(),
            (-1f32).sin(),
            4096f32.sin(),
            (-4096f32).sin(),
            (i16::MIN as f32).sin(),
            (i16::MAX as f32).sin()
        ]
    );
}
