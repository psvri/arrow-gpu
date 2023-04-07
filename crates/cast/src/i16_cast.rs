use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::Cast;

const I16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/cast_i32.wgsl")
);

#[async_trait]
impl Cast<Int32ArrayGPU> for Int16ArrayGPU {
    type Output = Int32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                2,
                I16_CAST_I32_SHADER,
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

    test_cast_op!(
        test_cast_i16_to_i32,
        Int16ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3],
        Int32Type,
        vec![0, 1, 2, 3, -1, -2, -3]
    );
}
