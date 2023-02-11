use async_trait::async_trait;

use crate::array::{i32_gpu::Int32ArrayGPU, ArrowArrayGPU, ArrowType};
#[async_trait]
pub trait Cast<T> {
    type Output;

    async fn cast(&self) -> Self::Output;
}

pub async fn cast_dyn(from: &ArrowArrayGPU, into: &ArrowType) -> ArrowArrayGPU {
    match (from, into) {
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowType::Int32Type) => {
            Cast::<Int32ArrayGPU>::cast(x).await.into()
        }
        (x, y) => panic!("Casting between {x:?} into {y:?} is not possible"),
    }
}
