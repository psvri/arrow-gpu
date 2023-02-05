use std::{any::Any, sync::Arc};

use async_trait::async_trait;

use crate::array::{f32_gpu::Float32ArrayGPU, ArrowArrayGPU, ArrowType};
#[async_trait]
pub trait Trigonometry {
    type Output;

    async fn sin(&self) -> Self::Output;
}

pub async fn sin_dyn(data: &dyn ArrowArrayGPU) -> Arc<dyn ArrowArrayGPU> {
    match data.get_data_type() {
        ArrowType::Float32Type => {
            let downcast_array = data
                .as_any()
                .downcast_ref::<Float32ArrayGPU>()
                .unwrap()
                .sin()
                .await;
            Arc::new(downcast_array)
        }
        ArrowType::UInt32Type => panic!("sin_dyn is not supported for this type"),
    }
}
