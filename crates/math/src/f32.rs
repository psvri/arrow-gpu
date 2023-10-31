use crate::{FloatMathUnaryType, MathUnaryType};
use arrow_gpu_array::array::*;
use std::sync::*;
use wgpu::Buffer;

const FLOATUNARY_SHADER: &str = include_str!("../compute_shaders/f32/floatunary.wgsl");
const UNARY_SHADER: &str = include_str!("../compute_shaders/f32/unary.wgsl");

impl MathUnaryType for f32 {
    type OutputType = Float32ArrayGPU;

    const SHADER: &'static str = UNARY_SHADER;
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

impl FloatMathUnaryType for f32 {
    type OutputType = Float32ArrayGPU;
    const SHADER: &'static str = FLOATUNARY_SHADER;
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
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_abs,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        abs,
        abs_dyn,
        vec![
            0.0f32.abs(),
            1.0f32.abs(),
            2.0f32.abs(),
            3.0f32.abs(),
            -1.0f32.abs(),
            -2.0f32.abs(),
            -3.0f32.abs()
        ]
    );

    test_unary_op_float!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_sqrt,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        sqrt,
        sqrt_dyn,
        vec![
            0.0f32.sqrt(),
            1.0f32.sqrt(),
            2.0f32.sqrt(),
            3.0f32.sqrt(),
            (-1.0f32).sqrt(),
            (-2.0f32).sqrt(),
            (-3.0f32).sqrt()
        ]
    );

    test_unary_op_float!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_exp,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        exp,
        exp_dyn,
        vec![
            0.0f32.exp(),
            1.0f32.exp(),
            2.0f32.exp(),
            3.0f32.exp(),
            (-1.0f32).exp(),
            (-2.0f32).exp(),
            (-3.0f32).exp()
        ]
    );

    test_unary_op_float!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_exp2,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        exp2,
        exp2_dyn,
        vec![
            0.0f32.exp2(),
            1.0f32.exp2(),
            2.0f32.exp2(),
            3.0f32.exp2(),
            (-1.0f32).exp2(),
            (-2.0f32).exp2(),
            (-3.0f32).exp2()
        ]
    );

    test_unary_op_float!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_log,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        log,
        log_dyn,
        vec![
            0.0f32.ln(),
            1.0f32.ln(),
            2.0f32.ln(),
            3.0f32.ln(),
            (-1.0f32).ln(),
            (-2.0f32).ln(),
            (-3.0f32).ln()
        ]
    );

    test_unary_op_float!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_f32_log2,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        log2,
        log2_dyn,
        vec![
            0.0f32.log2(),
            1.0f32.log2(),
            2.0f32.log2(),
            3.0f32.log2(),
            (-1.0f32).log2(),
            (-2.0f32).log2(),
            (-3.0f32).log2()
        ]
    );
}
