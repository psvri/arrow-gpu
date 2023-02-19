use crate::{kernels::arithmetic::*, ArrowErrorGPU};
use async_trait::async_trait;
use std::sync::Arc;

use super::{
    gpu_device::GpuDevice, primitive_array_gpu::*, u32_gpu::U32_SCALAR_SHADER, ArrowArrayGPU,
};

#[derive(Default, Debug)]
pub struct Date32Type {}

pub type Date32ArrayGPU = PrimitiveArrayGpu<Date32Type>;

#[async_trait]
impl ArrowScalarAdd<Date32ArrayGPU> for Date32ArrayGPU {
    type Output = Self;

    async fn add_scalar(&self, value: &Date32ArrayGPU) -> Self::Output {
        let new_buffer = self
            .gpu_device
            .apply_scalar_function(
                &self.data,
                &value.data,
                self.data.size(),
                4,
                U32_SCALAR_SHADER,
                "u32_add",
            )
            .await;

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: self.null_buffer.clone(),
        }
    }
}

impl From<Date32ArrayGPU> for ArrowArrayGPU {
    fn from(val: Date32ArrayGPU) -> Self {
        ArrowArrayGPU::Date32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Date32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Date32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Int32ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::kernels::broadcast::*;
    use crate::{
        array::primitive_array_gpu::test::{test_broadcast, test_scalar_op},
        kernels::arithmetic::add_scalar_dyn,
    };

    test_scalar_op!(
        test_add_date32_scalar_date32,
        Date32ArrayGPU,
        Date32ArrayGPU,
        vec![0, 1, 2, 3, 4],
        add_scalar,
        add_scalar_dyn,
        100i32,
        vec![100, 101, 102, 103, 104]
    );

    test_broadcast!(test_broadcast_date32, Date32ArrayGPU, 1);
}
