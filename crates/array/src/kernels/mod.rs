use crate::array::ArrowArrayGPU;

pub mod aggregate;
pub mod broadcast;

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

#[derive(Debug)]
pub enum Operand {
    Scalar(ScalarValue),
    Array(ArrowArrayGPU),
}
