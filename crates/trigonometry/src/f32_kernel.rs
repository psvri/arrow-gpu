use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;
use wgpu::Buffer;

use crate::{HyperbolicType, TrigonometricType};

const HYPERBOLIC_SHADER: &str = include_str!("../compute_shaders/f32/hyperbolic.wgsl");
const TRIGONOMETRY_SHADER: &str = include_str!("../compute_shaders/f32/trigonometry.wgsl");

impl HyperbolicType for f32 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = HYPERBOLIC_SHADER;
    const TYPE_STR: &'static str = "f32";
    const BUFFER_SIZE_MULTIPLIER: u64 = 1;

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

impl TrigonometricType for f32 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = TRIGONOMETRY_SHADER;
    const TYPE_STR: &'static str = "f32";
    const BUFFER_SIZE_MULTIPLIER: u64 = 1;

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
        test_f32_sinh,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        sinh,
        sinh_dyn,
        [
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
        test_f32_cos,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0],
        cos,
        cos_dyn,
        [
            0.0f32.cos(),
            1.0f32.cos(),
            2.0f32.cos(),
            3.0f32.cos(),
            (-1.0f32).cos(),
            (-2.0f32).cos()
        ]
    );

    test_unary_op_float!(
        test_f32_sin,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        sin,
        sin_dyn,
        [
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            -1.0f32.sin(),
            -2.0f32.sin(),
            -3.0f32.sin()
        ]
    );

    test_unary_op_float!(
        test_f32_acos,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        acos,
        acos_dyn,
        [
            0.0f32.acos(),
            1.0f32.acos(),
            2.0f32.acos(),
            3.0f32.acos(),
            (-1.0f32).acos(),
            (-2.0f32).acos(),
            (-3.0f32).acos()
        ]
    );
}
