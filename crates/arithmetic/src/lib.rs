pub(crate) mod f32;
pub(crate) mod i32;
pub(crate) mod kernels;
pub(crate) mod u16;
pub(crate) mod u32;

pub use kernels::*;

macro_rules! impl_arithmetic_op {
    ($trait_name: ident, $array_type:ident, $trait_function: ident, $ty: ident, $ty_size: expr, $shader: ident, $entry_point: expr) => {
        #[async_trait]
        impl<T> $trait_name<PrimitiveArrayGpu<T>> for $ty
        where
            T: $array_type + ArrowPrimitiveType,
        {
            type Output = Self;

            async fn $trait_function(&self, value: &PrimitiveArrayGpu<T>) -> Self::Output {
                let new_buffer = self
                    .gpu_device
                    .apply_scalar_function(
                        &self.data,
                        &value.data,
                        self.data.size(),
                        $ty_size,
                        $shader,
                        $entry_point,
                    )
                    .await;

                Self {
                    data: Arc::new(new_buffer),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer).await,
                }
            }
        }
    };
}

pub(crate) use impl_arithmetic_op;

macro_rules! impl_arithmetic_array_op {
    ($trait_name: ident, $array_type:ident, $trait_function: ident, $ty: ident, $ty_size: expr, $shader: ident, $entry_point: expr) => {
        #[async_trait]
        impl<T> $trait_name<PrimitiveArrayGpu<T>> for $ty
        where
            T: $array_type + ArrowPrimitiveType,
        {
            type Output = Self;

            async fn $trait_function(&self, value: &PrimitiveArrayGpu<T>) -> Self::Output {
                assert!(Arc::ptr_eq(&self.gpu_device, &value.gpu_device));
                let new_data_buffer = self
                    .gpu_device
                    .apply_binary_function(&self.data, &value.data, $ty_size, $shader, $entry_point)
                    .await;
                let new_null_buffer =
                    NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &value.null_buffer)
                        .await;

                Self {
                    data: Arc::new(new_data_buffer),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer: new_null_buffer,
                }
            }
        }
    };
}

pub(crate) use impl_arithmetic_array_op;
