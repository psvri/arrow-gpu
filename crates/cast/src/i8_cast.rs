use std::sync::Arc;

use arrow_gpu_array::array::{i32_gpu::Int32ArrayGPU, i8_gpu::Int8ArrayGPU};
use async_trait::async_trait;

use crate::Cast;

const I8_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_i32.wgsl")
);

#[async_trait]
impl Cast<Int32ArrayGPU> for Int8ArrayGPU {
    type Output = Int32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                I8_CAST_I32_SHADER,
                "cast_i32",
            )
            .await;

        Int32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;
    use arrow_gpu_array::array::gpu_device::GpuDevice;
    use arrow_gpu_array::array::ArrowType;

    test_cast_op!(
        test_cast_i8_to_i32,
        Int8ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        cast,
        Int32Type,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7]
    );
}
