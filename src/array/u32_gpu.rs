use crate::{kernels::arithmetic::*, ArrowErrorGPU};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

use super::{
    gpu_ops::u32_ops::*, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowType, GpuDevice,
    NullBitBufferGpu,
};

pub type UInt32ArrayGPU = PrimitiveArrayGpu<u32>;

impl_add_scalar_trait!(u32, add_scalar);
impl_sub_scalar_trait!(u32, sub_scalar);
impl_mul_scalar_trait!(u32, mul_scalar);
impl_div_scalar_trait!(u32, div_scalar);

impl_array_add_trait!(UInt32ArrayGPU, UInt32ArrayGPU, add_array_u32);

impl Into<ArrowArrayGPU> for UInt32ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::UInt32ArrayGPU(self)
    }
}

impl TryFrom<ArrowArrayGPU> for UInt32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::UInt32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into UInt32ArrayGPU",
                x
            ))),
        }
    }
}

impl UInt32ArrayGPU {
    pub async fn braodcast(value: u32, len: usize, gpu_device: Arc<GpuDevice>) -> Self {
        let data = Arc::new(braodcast_u32(&gpu_device, value, len.try_into().unwrap()).await);
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

impl ArrowArray for UInt32ArrayGPU {
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
        test_add_u32_array_u32,
        UInt32ArrayGPU,
        vec![Some(0u32), Some(1), None, None, Some(4)],
        vec![Some(1u32), Some(2), None, Some(4), None],
        vec![Some(1), Some(3), None, None, None]
    );

    test_scalar_op!(
        test_add_u32_scalar_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100u32,
        vec![100, 101, 102, 103, 104]
    );

    test_scalar_op!(
        test_sub_u32_scalar_u32,
        u32,
        vec![0, 100, 200, 3, 104],
        sub_scalar,
        100,
        vec![u32::MAX - 99, 0, 100, u32::MAX - 96, 4]
    );

    test_scalar_op!(
        test_mul_u32_scalar_u32,
        u32,
        vec![0, u32::MAX, 2, 3, 4],
        mul_scalar,
        100,
        vec![0, u32::MAX - 99, 200, 300, 400]
    );

    test_scalar_op!(
        test_div_u32_scalar_u32,
        u32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        100,
        vec![0, 0, 1, 2, 4]
    );

    test_scalar_op!(
        test_div_by_zero_u32_scalar_u32,
        u32,
        vec![0, 1, 100, 260, 450],
        div_scalar,
        0,
        vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
    );

    test_broadcast!(test_braodcast_u32, u32, 1);
}
