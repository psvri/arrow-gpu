use crate::array::ArrowArrayGPU;

pub mod broadcast;

/// Enum of scalar values used in kernels
#[derive(Debug)]
pub enum ScalarValue {
    F32(f32),
    U32(u32),
    U16(u16),
    U8(u8),
    I32(i32),
    I16(i16),
    I8(i8),
    BOOL(bool),
}

/// Enum of operands
#[derive(Debug)]
pub enum Operand {
    Scalar(ScalarValue),
    Array(ArrowArrayGPU),
}
