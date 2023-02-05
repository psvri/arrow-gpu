use std::{any::Any, sync::Arc};

use async_trait::async_trait;

use crate::array::f32_gpu::Float32ArrayGPU;
#[async_trait]
pub trait Trigonometry {
    type Output;

    async fn sin(&self) -> Self::Output;
}

pub async fn sin_dyn(data: &dyn Any) -> Arc<dyn Any> {
    if (&*data).is::<f32>() {
        Arc::new((*data.downcast_ref::<f32>().unwrap()).sin())
    } else if (&*data).is::<Float32ArrayGPU>() {
        Arc::new(
            (*data.downcast_ref::<Float32ArrayGPU>().unwrap())
                .sin()
                .await,
        )
    } else {
        panic!("Operation for this type is not yet implemented/unsupported ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sin_dyn() {
        assert_eq!(
            *(sin_dyn(&1.0f32).await).downcast_ref::<f32>().unwrap(),
            1.0f32.sin()
        )
    }
}
