use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::Cast;

const I16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/cast_i32.wgsl")
);
const I16_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../compute_shaders/i16/cast_f32.wgsl")
);

#[async_trait]
impl Cast<Int32ArrayGPU> for Int16ArrayGPU {
    async fn cast(&self) -> Int32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            I16_CAST_I32_SHADER,
            "cast_i32",
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
impl Cast<UInt32ArrayGPU> for Int16ArrayGPU {
    async fn cast(&self) -> UInt32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            I16_CAST_I32_SHADER,
            "cast_i32",
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
impl Cast<Float32ArrayGPU> for Int16ArrayGPU {
    async fn cast(&self) -> Float32ArrayGPU {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            self.data.size() * 2,
            2,
            I16_CAST_F32_SHADER,
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

#[async_trait]
impl Cast<UInt16ArrayGPU> for Int16ArrayGPU {
    async fn cast(&self) -> UInt16ArrayGPU {
        UInt16ArrayGPU {
            data: Arc::new(self.gpu_device.clone_buffer(&self.data)),
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
        test_cast_i16_to_i32,
        Int16ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, -1, 5713, -5713, i16::MIN, i16::MAX],
        Int32Type,
        vec![0, 1, -1, 5713, -5713, i16::MIN as i32, i16::MAX as i32]
    );

    test_cast_op!(
        test_cast_i16_to_u16,
        Int16ArrayGPU,
        UInt16ArrayGPU,
        vec![0, 1, -1, 5713, -5713, i16::MIN, i16::MAX],
        UInt16Type,
        vec![
            0,
            1,
            (-1i16) as u16,
            5713,
            (-5713i16) as u16,
            i16::MIN as u16,
            i16::MAX as u16
        ]
    );

    test_cast_op!(
        test_cast_i16_to_u32,
        Int16ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, -1, 5713, -5713, i16::MIN, i16::MAX],
        UInt32Type,
        vec![
            0,
            1,
            (-1i16) as u32,
            5713,
            (-5713i16) as u32,
            i16::MIN as u32,
            i16::MAX as u32
        ]
    );

    test_cast_op!(
        test_cast_i16_to_f32,
        Int16ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, -1, 5713, -5713, i16::MIN, i16::MAX],
        Float32Type,
        vec![
            0.0,
            1.0,
            (-1i16) as f32,
            5713.0,
            (-5713i16) as f32,
            i16::MIN as f32,
            i16::MAX as f32
        ]
    );
}
