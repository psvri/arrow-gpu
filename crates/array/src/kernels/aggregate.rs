pub trait ArrowSum {
    type Output;

    fn sum(&self) -> Self::Output;
}
