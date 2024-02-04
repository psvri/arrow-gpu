use std::sync::Arc;

use arrow_gpu_array::array::{types::*, *};
use arrow_gpu_array::gpu_utils::*;

use crate::impl_arithmetic_op;
use crate::*;

const F32_SCALAR_SHADER: &str = include_str!("../compute_shaders/f32/scalar.wgsl");
const F32_ARRAY_SHADER: &str = include_str!("../compute_shaders/f32/array.wgsl");
const F32_NEG_SHADER: &str = include_str!("../compute_shaders/f32/neg.wgsl");

impl Sum32Bit for f32 {
    const SHADER: &'static str = include_str!("../compute_shaders/f32/aggregate.wgsl");
}

impl_arithmetic_op!(
    ArrowScalarAdd,
    Float32Type,
    add_scalar_op,
    Float32ArrayGPU,
    F32_SCALAR_SHADER,
    "f32_add"
);

impl_arithmetic_op!(
    ArrowScalarSub,
    Float32Type,
    sub_scalar_op,
    Float32ArrayGPU,
    F32_SCALAR_SHADER,
    "f32_sub"
);

impl_arithmetic_op!(
    ArrowScalarMul,
    Float32Type,
    mul_scalar_op,
    Float32ArrayGPU,
    F32_SCALAR_SHADER,
    "f32_mul"
);

impl_arithmetic_op!(
    ArrowScalarDiv,
    Float32Type,
    div_scalar_op,
    Float32ArrayGPU,
    F32_SCALAR_SHADER,
    "f32_div"
);

impl_arithmetic_op!(
    ArrowScalarRem,
    Float32Type,
    rem_scalar_op,
    Float32ArrayGPU,
    F32_SCALAR_SHADER,
    "f32_rem"
);

impl_arithmetic_array_op!(
    ArrowAdd,
    Float32Type,
    add_op,
    Float32ArrayGPU,
    F32_ARRAY_SHADER,
    "add_f32"
);

impl_arithmetic_array_op!(
    ArrowSub,
    Float32Type,
    sub_op,
    Float32ArrayGPU,
    F32_ARRAY_SHADER,
    "sub_f32"
);

impl_arithmetic_array_op!(
    ArrowMul,
    Float32Type,
    mul_op,
    Float32ArrayGPU,
    F32_ARRAY_SHADER,
    "mul_f32"
);

impl_arithmetic_array_op!(
    ArrowDiv,
    Float32Type,
    div_op,
    Float32ArrayGPU,
    F32_ARRAY_SHADER,
    "div_f32"
);

impl NegUnaryType for f32 {
    type OutputType = Float32ArrayGPU;

    const SHADER: &'static str = F32_NEG_SHADER;

    const BUFFER_SIZE_MULTIPLIER: u64 = 1;

    fn create_new(
        data: Arc<wgpu::Buffer>,
        gpu_device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType {
        Float32ArrayGPU {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rem_scalar_dyn;
    use crate::test::test_sum;
    use arrow_gpu_test_macros::*;

    test_float_scalar_op!(
        test_rem_f32_scalar_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, 104.0],
        rem_scalar,
        rem_scalar_dyn,
        100.0,
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );

    test_scalar_op!(
        test_add_f32_scalar_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        add_scalar,
        add_dyn,
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );

    test_scalar_op!(
        test_div_f32_scalar_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        div_scalar,
        div_scalar_dyn,
        100.0,
        vec![0.0, 0.01, 0.02, 0.03, 0.04]
    );

    test_scalar_op!(
        test_mul_f32_scalar_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        mul_scalar,
        mul_dyn,
        100.0,
        vec![0.0, 100.0, 200.0, 300.0, 400.0]
    );

    test_scalar_op!(
        test_sub_f32_scalar_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0],
        sub_scalar,
        sub_scalar_dyn,
        100.0,
        vec![-100.0, -99.0, -98.0, -97.0, -96.0]
    );

    #[test]
    #[cfg_attr(
        target_os = "windows",
        ignore = "Not passing in CI but passes in local 🤔"
    )]
    fn test_large_f32_array() {
        let device = Arc::new(GpuDevice::new());
        let gpu_array = Float32ArrayGPU::from_slice(
            &(0..1024 * 1024 * 10)
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            device.clone(),
        );
        let values_array = Float32ArrayGPU::from_slice(&vec![100.0], device);
        let new_gpu_array = gpu_array.add_scalar(&values_array);
        for (index, value) in new_gpu_array.raw_values().unwrap().into_iter().enumerate() {
            assert_eq!((index as f32) + 100.0, value);
        }
    }

    test_array_op!(
        test_add_f32_array_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        add,
        add_dyn,
        vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
        vec![Some(1.0), Some(2.0), None, Some(4.0), None],
        vec![Some(1.0), Some(3.0), None, None, None]
    );

    test_array_op!(
        test_sub_f32_array_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        sub,
        sub_dyn,
        vec![Some(0.0), Some(1.0), None, None, Some(4.0), Some(10.0)],
        vec![Some(1.0), Some(2.0), None, Some(4.0), None, Some(0.0)],
        vec![Some(-1.0), Some(-1.0), None, None, None, Some(10.0)]
    );

    test_array_op!(
        test_mul_f32_array_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        mul,
        mul_dyn,
        vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
        vec![Some(1.0), Some(2.0), None, Some(4.0), None],
        vec![Some(0.0), Some(2.0), None, None, None]
    );

    test_array_op!(
        test_div_f32_array_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        div,
        div_dyn,
        vec![Some(0.0), Some(1.0), None, None, Some(4.0)],
        vec![Some(1.0), Some(2.0), None, Some(4.0), None],
        vec![Some(0.0), Some(0.5), None, None, None]
    );

    test_unary_op_float!(
        test_f32_exp2,
        Float32ArrayGPU,
        Float32ArrayGPU,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        neg,
        neg_dyn,
        vec![0.0, -1.0, -2.0, -3.0, 1.0, 2.0, 3.0]
    );

    test_sum!(
        test_f32_sum,
        Float32ArrayGPU,
        5.0,
        256 * 256,
        256.0 * 256.0 * 5.0
    );

    test_sum!(
        test_f32_sum_large,
        Float32ArrayGPU,
        5.0,
        4 * 1024 * 1024,
        4.0 * 1024.0 * 1024.0 * 5.0
    );
}
