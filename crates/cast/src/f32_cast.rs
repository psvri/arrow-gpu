use arrow_gpu_array::array::{Float32ArrayGPU, NullBitBufferGpu, UInt8ArrayGPU};
use arrow_gpu_array::gpu_utils::*;
use std::sync::Arc;

use crate::Cast;

const F32_CAST_U8_SHADER: &str = include_str!("../compute_shaders/f32/cast_u8.wgsl");

impl Cast<UInt8ArrayGPU> for Float32ArrayGPU {
    fn cast_op(&self, pipeline: &mut ArrowComputePipeline) -> UInt8ArrayGPU {
        let dispatch_size = self.data.size().div_ceil(16).div_ceil(256) as u32;

        let new_buffer = pipeline.apply_unary_function(
            &self.data,
            (self.data.size() / 4).next_multiple_of(4),
            F32_CAST_U8_SHADER,
            "cast_u8",
            dispatch_size,
        );

        let null_buffer =
            NullBitBufferGpu::clone_null_bit_buffer_pass(&self.null_buffer, &mut pipeline.encoder);

        UInt8ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer,
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
