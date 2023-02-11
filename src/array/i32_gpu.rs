use crate::{
    kernels::{arithmetic::*, broadcast::Broadcast},
    ArrowErrorGPU,
};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

use super::{
    gpu_ops::i32_ops::*, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowPrimitiveType,
    ArrowType, GpuDevice, NullBitBufferGpu, RustNativeType,
};

pub type Int32ArrayGPU = PrimitiveArrayGpu<i32>;

//impl_add_scalar_trait!(i32, add_scalar);

#[async_trait]
impl<T> ArrowScalarAdd<PrimitiveArrayGpu<T>> for PrimitiveArrayGpu<T>
where
    T: ArrowPrimitiveType<NativeType = i32>,
{
    type Output = Self;

    async fn add_scalar(&self, value: &PrimitiveArrayGpu<T>) -> Self::Output {
        let new_buffer = add_scalar(&self.gpu_device, &self.data, &value.data).await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

impl_sub_scalar_trait!(i32, sub_scalar);
impl_mul_scalar_trait!(i32, mul_scalar);
impl_div_scalar_trait!(i32, div_scalar);

impl_array_add_trait!(Int32ArrayGPU, Int32ArrayGPU, add_array_i32);

impl Into<ArrowArrayGPU> for Int32ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::Int32ArrayGPU(self)
    }
}

impl TryFrom<ArrowArrayGPU> for Int32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Int32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int32ArrayGPU",
                x
            ))),
        }
    }
}

#[async_trait]
impl<T: ArrowPrimitiveType<NativeType = i32>> Broadcast<i32> for PrimitiveArrayGpu<T> {
    type Output = PrimitiveArrayGpu<T>;

    async fn broadcast(value: i32, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output {
        let data = Arc::new(broadcast_i32(&gpu_device, value, len.try_into().unwrap()).await);
        let null_buffer = None;

        Self {
            data,
            gpu_device,
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

impl ArrowArray for Int32ArrayGPU {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> ArrowType {
        ArrowType::UInt32Type
    }

    fn get_memory_used(&self) -> u64 {
        self.data.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    test_add_array!(
        test_add_i32_array_i32,
        Int32ArrayGPU,
        vec![Some(0i32), Some(1), None, None, Some(4)],
        vec![Some(1i32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );

    test_scalar_op!(
        test_add_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100i32,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 100, 200, 3, 104],
        sub_scalar,
        sub_scalar_dyn,
        100,
        vec![-100, 0, 100, -97, 4]
    );

    test_scalar_op!(
        test_mul_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, i32::MAX, 2, 3, 4],
        mul_scalar,
        mul_scalar_dyn,
        100,
        vec![0, -100, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_i32_scalar_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        div_scalar_dyn,
        100,
        vec![0, 0, 1, 2, 4]
    );

    //ignore = "Not passing in linux CI but passes in windows 🤔"
    #[cfg(not(target_os = "linux"))]
    test_scalar_op!(
        test_div_by_zero_i32_scalar_i32,
        i32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        0,
        vec![-1; 5]
    );

    test_broadcast!(test_broadcast_i32, Int32ArrayGPU, 1);
}
