use arrow_gpu_array::array::*;
use async_trait::async_trait;

use crate::{
    u32::{U32_LOGICAL_SHADER, U32_NOT_SHADER},
    *,
};

impl LogicalType for BooleanArrayGPU {
    const SHADER: &'static str = U32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = "";
    const NOT_SHADER: &'static str = U32_NOT_SHADER;
}

#[async_trait]
impl Logical for BooleanArrayGPU {
    async fn bitwise_and(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(&self.data, &operand.data, 4, Self::SHADER, AND_ENTRY_POINT)
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_or(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(&self.data, &operand.data, 4, Self::SHADER, OR_ENTRY_POINT)
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_xor(&self, operand: &Self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(&self.data, &operand.data, 4, Self::SHADER, XOR_ENTRY_POINT)
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_not(&self) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(
                &self.data,
                self.data.size(),
                4,
                Self::NOT_SHADER,
                NOT_ENTRY_POINT,
            )
            .await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
        }
    }

    async fn bitwise_shl(&self, operand: &UInt32ArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                4,
                Self::SHIFT_SHADER,
                SHIFT_LEFT_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    async fn bitwise_shr(&self, operand: &UInt32ArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_binary_function(
                &self.data,
                &operand.data,
                4,
                Self::SHIFT_SHADER,
                SHIFT_RIGHT_ENTRY_POINT,
            )
            .await;
        let new_null_buffer =
            NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &operand.null_buffer).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_test_macros::test_array_op;

    test_array_op!(
        test_bitwise_and_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_or_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![Some(true), Some(true), Some(true), Some(false), None, None]
    );

    test_array_op!(
        test_bitwise_xor_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![Some(false), Some(true), Some(true), Some(false), None, None]
    );
}
