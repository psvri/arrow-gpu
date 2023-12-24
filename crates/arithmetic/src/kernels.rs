use std::sync::Arc;

use arrow_gpu_array::array::{
    ArrowArrayGPU, ArrowPrimitiveType, GpuDevice, NullBitBufferGpu, PrimitiveArrayGpu,
};
use wgpu::Buffer;

pub trait ArrowScalarAdd<Rhs> {
    type Output;

    fn add_scalar(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowScalarSub<Rhs> {
    type Output;

    fn sub_scalar(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowScalarMul<Rhs> {
    type Output;

    fn mul_scalar(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowScalarDiv<Rhs> {
    type Output;

    fn div_scalar(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowScalarRem<Rhs> {
    type Output;

    fn rem_scalar(&self, value: &Rhs) -> Self::Output;
}

pub fn add_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(arr), ArrowArrayGPU::UInt16ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn sub_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.sub_scalar(scalar).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn mul_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.mul_scalar(scalar).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn div_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.div_scalar(scalar).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn rem_scalar_dyn(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (data, value) {
        (ArrowArrayGPU::Float32ArrayGPU(arr), ArrowArrayGPU::Float32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr), ArrowArrayGPU::UInt32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        /*(ArrowArrayGPU::UInt16ArrayGPU(arr), ArrowArrayGPU::UInt16ArrayGPU(scalar)) => {
            arr.add_scalar(scalar).into()
        }*/
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Date32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr), ArrowArrayGPU::Int32ArrayGPU(scalar)) => {
            arr.rem_scalar(scalar).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub trait ArrowAdd<Rhs> {
    type Output;

    fn add(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowDiv<Rhs> {
    type Output;

    fn div(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowMul<Rhs> {
    type Output;

    fn mul(&self, value: &Rhs) -> Self::Output;
}

pub trait ArrowSub<Rhs> {
    type Output;

    fn sub(&self, value: &Rhs) -> Self::Output;
}

pub fn add_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(arr1), ArrowArrayGPU::UInt32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr1), ArrowArrayGPU::Int32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(arr1), ArrowArrayGPU::Date32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr1), ArrowArrayGPU::Date32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        (ArrowArrayGPU::Date32ArrayGPU(arr1), ArrowArrayGPU::Int32ArrayGPU(arr2)) => {
            arr1.add(arr2).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn sub_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.sub(arr2).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn mul_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.mul(arr2).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn div_array_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1, input2) {
        (ArrowArrayGPU::Float32ArrayGPU(arr1), ArrowArrayGPU::Float32ArrayGPU(arr2)) => {
            arr1.div(arr2).into()
        }
        _ => panic!("Operation not supported"),
    }
}

pub fn add_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => add_array_dyn(input1, input2),
        (_, 1) => add_scalar_dyn(input1, input2),
        (1, _) => add_scalar_dyn(input2, input1),
        _ => unreachable!(),
    }
}

pub fn sub_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => sub_array_dyn(input1, input2),
        (_, 1) => sub_scalar_dyn(input1, input2),
        (1, _) => sub_scalar_dyn(input2, input1),
        _ => unreachable!(),
    }
}

pub fn mul_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => mul_array_dyn(input1, input2),
        (_, 1) => mul_scalar_dyn(input1, input2),
        (1, _) => mul_scalar_dyn(input2, input1),
        _ => unreachable!(),
    }
}

pub fn div_dyn(input1: &ArrowArrayGPU, input2: &ArrowArrayGPU) -> ArrowArrayGPU {
    match (input1.len(), input2.len()) {
        (x, y) if (x == 1 && y == 1) || (x != 1 && y != 1) => div_array_dyn(input1, input2),
        (_, 1) => div_scalar_dyn(input1, input2),
        (1, _) => div_scalar_dyn(input2, input1),
        _ => unreachable!(),
    }
}

pub trait Neg {
    type OutputType;
    fn neg(&self) -> Self::OutputType;
}

pub trait NegUnaryType {
    type OutputType;
    const SHADER: &'static str;
    const BUFFER_SIZE_MULTIPLIER: u64;

    fn create_new(
        data: Arc<Buffer>,
        device: Arc<GpuDevice>,
        len: usize,
        null_buffer: Option<NullBitBufferGpu>,
    ) -> Self::OutputType;
}

impl<T: NegUnaryType + ArrowPrimitiveType> Neg for PrimitiveArrayGpu<T> {
    type OutputType = T::OutputType;

    fn neg(&self) -> Self::OutputType {
        let new_buffer = self.gpu_device.apply_unary_function(
            &self.data,
            &self.data.size() * <T as NegUnaryType>::BUFFER_SIZE_MULTIPLIER,
            <T as ArrowPrimitiveType>::ITEM_SIZE,
            T::SHADER,
            "neg",
        );
        let new_null_buffer = NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer);

        return <T as NegUnaryType>::create_new(
            Arc::new(new_buffer),
            self.gpu_device.clone(),
            self.len,
            new_null_buffer,
        );
    }
}

pub fn neg_dyn(input: &ArrowArrayGPU) -> ArrowArrayGPU {
    match input {
        ArrowArrayGPU::Float32ArrayGPU(x) => x.neg().into(),
        _ => panic!(
            "Operation negation not supported for type {:?}",
            input.get_dtype()
        ),
    }
}
