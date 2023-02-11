use crate::array::ArrowArrayGPU;

pub mod aggregate;
pub mod arithmetic;
pub mod broadcast;
pub mod cast;
pub mod trigonometry;

#[derive(Debug)]
pub enum ScalarValue {
    F32(f32),
    U32(u32),
    U16(u16),
}

#[derive(Debug)]
pub enum Operand {
    Scalar(ScalarValue),
    Array(ArrowArrayGPU),
}
