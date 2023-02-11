use crate::{kernels::arithmetic::*, ArrowErrorGPU};
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

use super::{
    gpu_ops::i32_ops::*, primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowType, GpuDevice,
    NullBitBufferGpu,
};

#[derive(Default, Debug)]
pub struct Date32Type {}

pub type Date32ArrayGPU = PrimitiveArrayGpu<Date32Type>;

impl Into<ArrowArrayGPU> for Date32ArrayGPU {
    fn into(self) -> ArrowArrayGPU {
        ArrowArrayGPU::Date32ArrayGPU(self)
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
