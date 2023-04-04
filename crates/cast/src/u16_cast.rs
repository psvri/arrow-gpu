use std::sync::Arc;

use arrow_gpu_array::array::{u16_gpu::UInt16ArrayGPU, u32_gpu::UInt32ArrayGPU};
use async_trait::async_trait;

use crate::Cast;

const U16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cast_u32.wgsl")
);

#[async_trait]
impl Cast<UInt32ArrayGPU> for UInt16ArrayGPU {
    type Output = UInt32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                2,
                U16_CAST_I32_SHADER,
                "cast_u32",
            )
            .await;

        UInt32ArrayGPU {
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
        test_cast_u16_to_u32,
        UInt16ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        UInt32Type,
        vec![0, 1, 2, 3, 4]
    );
}
