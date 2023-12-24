use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::Cast;

const U16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cast_u32.wgsl")
);
const U16_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cast_f32.wgsl")
);

#[async_trait]
impl Cast<Int16ArrayGPU> for UInt16ArrayGPU {
    async fn cast(&self) -> Int16ArrayGPU {
        Int16ArrayGPU {
            data: Arc::new(self.gpu_device.clone_buffer(&self.data)),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<Int32ArrayGPU> for UInt16ArrayGPU {
    async fn cast(&self) -> Int32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            U16_CAST_I32_SHADER,
            "cast_u32",
        );

        Int32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<UInt32ArrayGPU> for UInt16ArrayGPU {
    async fn cast(&self) -> UInt32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            U16_CAST_I32_SHADER,
            "cast_u32",
        );

        UInt32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<Float32ArrayGPU> for UInt16ArrayGPU {
    async fn cast(&self) -> Float32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            U16_CAST_F32_SHADER,
            "cast_f32",
        );

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;

    test_cast_op!(
        test_cast_u16_to_i16,
        UInt16ArrayGPU,
        Int16ArrayGPU,
        vec![0, 1, 2, 3, 4, u16::MAX, u16::MAX - 1, u16::MAX - 2],
        Int16Type,
        vec![0, 1, 2, 3, 4, -1, -2, -3]
    );

    test_cast_op!(
        test_cast_u16_to_i32,
        UInt16ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 4, u16::MAX, u16::MAX - 1, u16::MAX - 2],
        Int32Type,
        vec![
            0,
            1,
            2,
            3,
            4,
            u16::MAX as i32,
            (u16::MAX - 1) as i32,
            (u16::MAX - 2) as i32
        ]
    );

    test_cast_op!(
        test_cast_u16_to_u32,
        UInt16ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        UInt32Type,
        vec![0, 1, 2, 3, 4]
    );

    test_cast_op!(
        test_cast_u16_to_f32,
        UInt16ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 4, u16::MAX],
        Float32Type,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, u16::MAX as f32]
    );
}
