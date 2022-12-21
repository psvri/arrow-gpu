use async_trait::async_trait;

#[async_trait]
pub trait ArrowAdd<Rhs> {
    type Output;

    async fn add(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowDiv<Rhs> {
    type Output;

    async fn div(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowMul<Rhs> {
    type Output;

    async fn mul(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowSub<Rhs> {
    type Output;

    async fn sub(&self, value: &Rhs) -> Self::Output;
}

#[async_trait]
pub trait ArrowAddAssign<Rhs> {
    async fn add_assign(&mut self, value: &Rhs);
}