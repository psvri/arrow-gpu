use crate::ArrowErrorGPU;

use super::{primitive_array_gpu::*, ArrowArrayGPU};

#[derive(Default, Debug)]
pub struct Date32Type {}

pub type Date32ArrayGPU = PrimitiveArrayGpu<Date32Type>;

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
                "could not cast {:?} into Date32ArrayGPU",
                x
            ))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::array::primitive_array_gpu::test::test_broadcast;
    use crate::array::*;
    use crate::kernels::broadcast::*;
    use std::sync::Arc;

    test_broadcast!(test_broadcast_date32, Date32ArrayGPU, 1);
}
