use async_trait::async_trait;

use crate::array::ArrowArrayGPU;

#[async_trait]
pub trait ArrowScalarAdd<Rhs> {
    type Output;

    async fn add_scalar(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowScalarSub<Rhs> {
    type Output;

    async fn sub_scalar(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowScalarMul<Rhs> {
    type Output;

    async fn mul_scalar(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowScalarDiv<Rhs> {
    type Output;

    async fn div_scalar(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowAdd<Rhs> {
    type Output;

    async fn add(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowDiv<Rhs> {
    type Output;

    async fn div(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowMul<Rhs> {
    type Output;

    async fn mul(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowSub<Rhs> {
    type Output;

    async fn sub(&self, value: &Rhs) -> Self::Output;
}

pub async fn add_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}
