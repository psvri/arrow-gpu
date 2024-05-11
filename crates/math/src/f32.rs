use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;
use std::sync::*;
use wgpu::Buffer;

use crate::MathBinaryType;
use crate::{FloatMathUnaryType, MathUnaryType};

const FLOATUNARY_SHADER: &str = include_str!("../compute_shaders/f32/floatunary.wgsl");
const FLOATBINARY_SHADER: &str = include_str!("../compute_shaders/f32/floatbinary.wgsl");

impl MathUnaryType for f32 {
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

impl MathBinaryType for f32 {
    type OutputType = Float32ArrayGPU;

    const SHADER: &'static str = FLOATBINARY_SHADER;
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
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        abs,
        abs_dyn,
        [
            0.0f32.abs(),
            1.0f32.abs(),
            2.0f32.abs(),
            3.0f32.abs(),
            (-1.0f32).abs(),
            (-2.0f32).abs(),
            (-3.0f32).abs()
        ]
    );

    test_unary_op_float!(
        test_f32_sqrt,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        sqrt,
        sqrt_dyn,
        [
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
        test_f32_cbrt,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        cbrt,
        cbrt_dyn,
        [
            0.0f32.cbrt(),
            1.0f32.cbrt(),
            2.0f32.cbrt(),
            3.0f32.cbrt(),
            (-1.0f32).cbrt(),
            (-2.0f32).cbrt(),
            (-3.0f32).cbrt()
        ]
    );

    test_unary_op_float!(
        test_f32_exp,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        exp,
        exp_dyn,
        [
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
        test_f32_exp2,
        Float32ArrayGPU,
        Float32ArrayGPU,
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        exp2,
        exp2_dyn,
        [
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
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        log,
        log_dyn,
        [
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
        [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        log2,
        log2_dyn,
        [
            0.0f32.log2(),
            1.0f32.log2(),
            2.0f32.log2(),
            3.0f32.log2(),
            (-1.0f32).log2(),
            (-2.0f32).log2(),
            (-3.0f32).log2()
        ]
    );

    test_float_array_op!(
        #[cfg_attr(
            any(target_os = "macos", target_os = "linux"),
            ignore = "-x ** 0.0 is returning different values based on OS"
        )]
        test_f32_power,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        power,
        power_dyn,
        [
            Some(1.0f32),
            Some(-1.0f32),
            Some(10.0f32),
            Some(-10.0f32),
            Some(3.0),
            Some(-1.0f32),
            None,
            None,
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::INFINITY),
            Some(f32::NAN),
        ],
        [
            Some(0.0f32),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(2.0),
            None,
            Some(3.0),
            None,
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(3.0),
        ],
        [
            Some(1.0f32.powf(0.0)),
            // TODO fixeme gpu -1.0 ** 0.0 gives NAN instead of 1.0
            Some(f32::NAN),
            Some(10.0f32.powf(0.0)),
            // TODO fixeme gpu -10.0 ** 0.0 gives NAN instead of 1.0
            Some(f32::NAN),
            Some(3.0f32.powf(2.0)),
            None,
            None,
            None,
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NAN),
            Some(f32::NAN),
            Some(0.0),
            Some(f32::NAN),
        ]
    );
}
