use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::Cast;

const U8_CAST_U16_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/cast_u16.wgsl")
);
const U8_CAST_U32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/cast_u32.wgsl")
);
const U8_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/cast_f32.wgsl")
);

#[async_trait]
impl Cast<Int8ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> Int8ArrayGPU {
        let new_buffer = self.gpu_device.clone_buffer(&self.data);

        Int8ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<Int16ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> Int16ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                1,
                U8_CAST_U16_SHADER,
                "cast_u16",
            )
            .await;

        Int16ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<Int32ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> Int32ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                U8_CAST_U32_SHADER,
                "cast_u32",
            )
            .await;

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
impl Cast<UInt16ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> UInt16ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                1,
                U8_CAST_U16_SHADER,
                "cast_u16",
            )
            .await;

        UInt16ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}

#[async_trait]
impl Cast<UInt32ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> UInt32ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                U8_CAST_U32_SHADER,
                "cast_u32",
            )
            .await;

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
impl Cast<Float32ArrayGPU> for UInt8ArrayGPU {

    async fn cast(&self) -> Float32ArrayGPU {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                U8_CAST_F32_SHADER,
                "cast_f32",
            )
            .await;

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
        test_cast_u8_to_i8,
        UInt8ArrayGPU,
        Int8ArrayGPU,
        vec![
            0,
            1,
            2,
            3,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 2,
            u8::MAX - 6,
            7
        ],
        Int8Type,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_u8_to_i16,
        UInt8ArrayGPU,
        Int16ArrayGPU,
        vec![0, 1, 2, 3, 255, 254, 253, 249, 7],
        Int16Type,
        vec![0, 1, 2, 3, 255, 254, 253, 249, 7]
    );

    test_cast_op!(
        test_cast_u8_to_i32,
        UInt8ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 255, 254, 253, 249, 7],
        Int32Type,
        vec![0, 1, 2, 3, 255, 254, 253, 249, 7]
    );

    test_cast_op!(
        test_cast_u8_to_u16,
        UInt8ArrayGPU,
        UInt16ArrayGPU,
        vec![0, 1, 2, 3, 255, 250, 7],
        UInt16Type,
        vec![0, 1, 2, 3, 255, 250, 7]
    );

    test_cast_op!(
        test_cast_u8_to_u32,
        UInt8ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 255, 250, 7],
        UInt32Type,
        vec![0, 1, 2, 3, 255, 250, 7]
    );

    test_cast_op!(
        test_cast_u8_to_f32,
        UInt8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, 255, 250, 7],
        Float32Type,
        vec![0.0, 1.0, 2.0, 3.0, 255.0, 250.0, 7.0]
    );
}
