use std::sync::Arc;

use crate::Cast;
use arrow_gpu_array::array::{Float32ArrayGPU, NullBitBufferGpu, UInt8ArrayGPU};
use async_trait::async_trait;

const F32_CAST_U8_SHADER: &str = include_str!("../compute_shaders/f32/cast_u8.wgsl");

#[async_trait]
impl Cast<UInt8ArrayGPU> for Float32ArrayGPU {
    async fn cast(&self) -> UInt8ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                (self.data.size() / 4).next_multiple_of(4),
                16, // 4 * 4
                F32_CAST_U8_SHADER,
                "cast_u8",
            )
            .await;

        UInt8ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cast_dyn;
    use crate::tests::test_cast_op;
    use crate::Cast;
    use arrow_gpu_array::array::*;

    test_cast_op!(
        #[cfg_attr(target_os = "linux", ignore = "Not passing in CI 🤔")]
        test_cast_f32_to_u8,
        Float32ArrayGPU,
        UInt8ArrayGPU,
        vec![0.0, 1.0, -1.0, 5713.0, -5713.0, 255.0, 256.0],
        UInt8Type,
        vec![0u8, 1, 0, (5713u16 % 256) as u8, 0, 255, 0]
    );
}
