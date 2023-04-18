use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, BooleanArrayGPU, GpuDevice, NullBitBufferGpu,
    PrimitiveArrayGpu,
};
use async_trait::async_trait;
use wgpu::Buffer;

pub(crate) mod i16;
pub(crate) mod i32;
pub(crate) mod i8;
pub(crate) mod u16;
pub(crate) mod u32;
pub(crate) mod u8;

#[async_trait]
pub trait Swizzle {
    // Selects self incase of true else selects from other.
    // None values in mask results in None
    async fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self;
}

pub trait SwizzleType {
    const MERGE_SHADER: &'static str;
}

#[async_trait]
impl<T: SwizzleType + ArrowPrimitiveType> Swizzle for PrimitiveArrayGpu<T> {
    async fn merge(&self, other: &Self, mask: &BooleanArrayGPU) -> Self {
        let new_buffer = self
            .gpu_device
            .apply_ternary_function(
                &self.data,
                &other.data,
                &mask.data,
                T::ITEM_SIZE,
                T::MERGE_SHADER,
                "merge",
            )
            .await;

        let bit_buffer: Buffer = merge_null_buffers(
            &self.gpu_device,
            &self.null_buffer.as_ref().unwrap().bit_buffer,
            &other.null_buffer.as_ref().unwrap().bit_buffer,
            &mask.data,
            &mask.null_buffer.as_ref().unwrap().bit_buffer,
        )
        .await;

        let new_null_buffer = NullBitBufferGpu {
            bit_buffer: Arc::new(bit_buffer),
            len: self.len,
            buffer_len: self.null_buffer.as_ref().unwrap().buffer_len,
            gpu_device: self.gpu_device.clone(),
        };

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: std::marker::PhantomData,
            len: self.len,
            null_buffer: Some(new_null_buffer),
        }
    }
}

pub async fn merge_null_buffers(
    device: &GpuDevice,
    operand_1_null_buffer: &Buffer,
    operand_2_null_buffer: &Buffer,
    mask: &Buffer,
    mask_null_buffer: &Buffer,
) -> Buffer {
    const SHADER: &str = include_str!("../compute_shaders/u32/merge_null_buffer.wgsl");

    let merged_buffer_1 = device
        .apply_binary_function(operand_1_null_buffer, mask, 4, SHADER, "merge_selected")
        .await;
    let merged_buffer_2 = device
        .apply_binary_function(operand_2_null_buffer, mask, 4, SHADER, "merge_not_selected")
        .await;
    let mut merged_buffer = device
        .apply_binary_function(&merged_buffer_1, &merged_buffer_2, 4, SHADER, "merge_or")
        .await;
    merged_buffer = device
        .apply_binary_function(&merged_buffer, mask_null_buffer, 4, SHADER, "merge_nulls")
        .await;
    merged_buffer
}

pub async fn merge_dyn(
    operand_1: &ArrowArrayGPU,
    operand_2: &ArrowArrayGPU,
    mask: &BooleanArrayGPU,
) -> ArrowArrayGPU {
    match (operand_1, operand_2) {
        (ArrowArrayGPU::Date32ArrayGPU(op1), ArrowArrayGPU::Date32ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(op1), ArrowArrayGPU::Int32ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(op1), ArrowArrayGPU::Int16ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(op1), ArrowArrayGPU::Int8ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(op1), ArrowArrayGPU::UInt32ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(op1), ArrowArrayGPU::UInt16ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(op1), ArrowArrayGPU::UInt8ArrayGPU(op2)) => {
            op1.merge(op2, mask).await.into()
        }
        _ => panic!(
            "Merge Operation not supported between {:?} and {:?}",
            operand_1, operand_2
        ),
    }
}

#[cfg(test)]
mod test {
    #[macro_export]
    macro_rules! test_merge_op {
        ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $mask: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(GpuDevice::new().await);
                let gpu_array_1 = $operand1_type::from_optional_vec(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_optional_vec(&$input_2, device.clone());
                let mask = BooleanArrayGPU::from_optional_vec(&$mask, device);
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2, &mask).await;
                assert_eq!(new_gpu_array.values().await, $output);
            }
        };
        ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr,  $mask: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(GpuDevice::new().await);
                let gpu_array_1 = $operand1_type::from_optional_vec(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_optional_vec(&$input_2, device.clone());
                let mask = BooleanArrayGPU::from_optional_vec(&$mask, device);
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2, &mask).await;
                assert_eq!(new_gpu_array.values().await, $output);

                let new_gpu_array =
                    $operation_dyn(&gpu_array_1.into(), &gpu_array_2.into(), &mask).await;
                let new_values = $output_type::try_from(new_gpu_array)
                    .unwrap()
                    .values()
                    .await;
                assert_eq!(new_values, $output);
            }
        };
    }
}
