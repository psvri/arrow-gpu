use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;
use wgpu::Buffer;

use crate::{HyperbolicType, TrigonometricType};

const HYPERBOLIC_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/hyperbolic.wgsl")
);

const TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/trigonometry.wgsl")
);

impl HyperbolicType for u8 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = HYPERBOLIC_SHADER;
    const TYPE_STR: &'static str = "u8";
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

impl TrigonometricType for u8 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = TRIGONOMETRY_SHADER;
    const TYPE_STR: &'static str = "u8";
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
        test_u8_sinh,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        sinh,
        sinh_dyn,
        vec![
            0.0f32.sinh(),
            1.0f32.sinh(),
            2.0f32.sinh(),
            3.0f32.sinh(),
            4.0f32.sinh()
        ]
    );

    test_unary_op_float!(
        test_u8_cos,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        cos,
        cos_dyn,
        vec![
            0.0f32.cos(),
            1.0f32.cos(),
            2.0f32.cos(),
            3.0f32.cos(),
            4.0f32.cos()
        ]
    );

    test_unary_op_float!(
        test_u8_sin,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 5],
        sin,
        sin_dyn,
        vec![
            0.0f32.sin(),
            1.0f32.sin(),
            2.0f32.sin(),
            3.0f32.sin(),
            5.0f32.sin()
        ]
    );
}
