use async_trait::async_trait;

#[async_trait]
pub trait ArrowSum {
    type Output;

    async fn sum(&self) -> Self::Output;
}
