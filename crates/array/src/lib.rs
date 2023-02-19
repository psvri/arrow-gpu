pub mod array;
pub mod kernels;
pub mod utils;

#[derive(Debug)]
pub enum ArrowErrorGPU {
    OperationNotSupported(String),
    CastingNotSupported(String),
}
