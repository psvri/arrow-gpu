use async_trait::async_trait;

use crate::array::ArrowArrayGPU;
#[async_trait]
pub trait Trigonometry {
    type Output;

    async fn sin(&self) -> Self::Output;
}

pub async fn sin_dyn(data: &ArrowArrayGPU) -> ArrowArrayGPU {
    match data {
        ArrowArrayGPU::Float32ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::UInt16ArrayGPU(arr) => arr.sin().await.into(),
        ArrowArrayGPU::UInt8ArrayGPU(arr) => arr.sin().await.into(),
        _ => panic!("Operation not supported"),
    }
}
