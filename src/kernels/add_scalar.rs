pub trait AddScalarKernel<Rhs> {
    type Output;

    fn add_scalar(&self, value: Rhs) -> Self::Output;
}