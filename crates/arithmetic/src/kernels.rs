use arrow_gpu_array::array::ArrowArrayGPU;
use async_trait::async_trait;

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
pub trait ArrowScalarRem<Rhs> {
    type Output;

    async fn rem_scalar(&self, value: &Rhs) -> Self::Output;
}

pub async fn add_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr), ArrowArrayGPU::UInt16ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn sub_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn mul_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn div_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn rem_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        /*(ArrowArrayGPU::UInt16ArrayGPU(arr), ArrowArrayGPU::UInt16ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).await.into()
        }*/
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).await.into()
        }
        _ => panic!("Operation not supported"),
    }
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

pub async fn add_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr1), ArrowArrayGPU::UInt32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr1), ArrowArrayGPU::Int32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr1), ArrowArrayGPU::Date32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr1), ArrowArrayGPU::Date32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr1), ArrowArrayGPU::Int32ArrayGPU(arr2)) => {
            arr1.add(arr2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn mul_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.mul(arr2).await.into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub async fn add_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => add_array_dyn(input1, input2).await,
        (_, 1) => add_scalar_dyn(input1, input2).await,
        (1, _) => add_scalar_dyn(input2, input1).await,
        _ => unreachable!(),
    }
}

pub async fn mul_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => mul_array_dyn(input1, input2).await,
        (_, 1) => mul_scalar_dyn(input1, input2).await,
        (1, _) => mul_scalar_dyn(input2, input1).await,
        _ => unreachable!(),
    }
}
