use async_trait::async_trait;

use crate::array::{ArrowArray, ArrowArrayGPU};

#[async_trait]
pub trait Logical<Rhs: ArrowArray>: ArrowArray {
    type Output;

    async fn bitwise_and(&self, operand: &Rhs) -> Self::Output;
    async fn bitwise_or(&self, operand: &Rhs) -> Self::Output;
}

pub async fn bitwise_and_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::Int32ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_and(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn bitwise_or_dyn(data_1: &ArrowArrayGPU, data_2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data_1, data_2) {
        (ArrowArrayGPU::Int32ArrayGPU(arr_1), ArrowArrayGPU::Int32ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr_1), ArrowArrayGPU::UInt32ArrayGPU(arr_2)) => {
            arr_1.bitwise_or(arr_2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}
