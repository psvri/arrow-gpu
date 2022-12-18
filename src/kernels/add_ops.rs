use async_trait::async_trait;

#[async_trait]
pub trait ArrowAddAssign<Rhs> {
    async fn add_assign(&mut self, value: Rhs);
}

#[async_trait]
pub trait ArrowAdd<Rhs> {
    type Output;

    async fn add(&mut self, value: Rhs) -> Self::Output;
}
