use std::sync::Arc;

use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::Cast;

const I8_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_i32.wgsl")
);
const I8_CAST_I16_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_i16.wgsl")
);
const I8_CAST_F32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../compute_shaders/i8/cast_f32.wgsl")
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
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[async_trait]
impl Cast<UInt32ArrayGPU> for Int8ArrayGPU {
    type Output = UInt32ArrayGPU;

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

        UInt32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[async_trait]
impl Cast<Int16ArrayGPU> for Int8ArrayGPU {
    type Output = Int16ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                1,
                I8_CAST_I16_SHADER,
                "cast_i16",
            )
            .await;

        Int16ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[async_trait]
impl Cast<UInt16ArrayGPU> for Int8ArrayGPU {
    type Output = UInt16ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 2,
                1,
                I8_CAST_I16_SHADER,
                "cast_i16",
            )
            .await;

        UInt16ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[async_trait]
impl Cast<UInt8ArrayGPU> for Int8ArrayGPU {
    type Output = UInt8ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self.gpu_device.clone_buffer(&self.data).await;

        UInt8ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[async_trait]
impl Cast<Float32ArrayGPU> for Int8ArrayGPU {
    type Output = Float32ArrayGPU;

    async fn cast(&self) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size() * 4,
                1,
                I8_CAST_F32_SHADER,
                "cast_f32",
            )
            .await;

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;

    test_cast_op!(
        test_cast_i8_to_i32,
        Int8ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        Int32Type,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_i8_to_u32,
        Int8ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt32Type,
        vec![
            0,
            1,
            2,
            3,
            u32::MAX,
            u32::MAX - 1,
            u32::MAX - 2,
            u32::MAX - 6,
            7
        ]
    );

    test_cast_op!(
        test_cast_i8_to_i16,
        Int8ArrayGPU,
        Int16ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        Int16Type,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7]
    );

    test_cast_op!(
        test_cast_i8_to_u16,
        Int8ArrayGPU,
        UInt16ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt16Type,
        vec![
            0,
            1,
            2,
            3,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 2,
            u16::MAX - 6,
            7
        ]
    );

    test_cast_op!(
        test_cast_i8_to_u8,
        Int8ArrayGPU,
        UInt8ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        UInt8Type,
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
        ]
    );

    test_cast_op!(
        test_cast_i8_to_f32,
        Int8ArrayGPU,
        Float32ArrayGPU,
        vec![0, 1, 2, 3, -1, -2, -3, -7, 7],
        Float32Type,
        vec![0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -7.0, 7.0]
    );
}
