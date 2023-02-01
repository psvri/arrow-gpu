use async_trait::async_trait;

#[async_trait]
pub trait Trigonometry {
    type Output;

    async fn sin(&self) -> Self::Output;
}
