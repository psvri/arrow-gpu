pub(crate) mod aggregate_kernels;
pub(crate) mod arithmetic_kernels;
pub(crate) mod f32;
pub(crate) mod i32;
pub(crate) mod u16;
pub(crate) mod u32;

pub use aggregate_kernels::*;
pub use arithmetic_kernels::*;

macro_rules! impl_arithmetic_op {
    ($trait_name: ident, $array_type:ident, $trait_function: ident, $ty: ident, $shader: ident, $entry_point: expr) => {
        impl<T> $trait_name<PrimitiveArrayGpu<T>> for $ty
        where
            T: $array_type + ArrowPrimitiveType,
        {
            type Output = Self;

            fn $trait_function(
                &self,
                value: &PrimitiveArrayGpu<T>,
                pipeline: &mut ArrowComputePipeline,
            ) -> Self::Output {
                let dispatch_size = self.data.size().div_ceil(T::ITEM_SIZE).div_ceil(256) as u32;

                let new_buffer = pipeline.apply_scalar_function(
                    &self.data,
                    &value.data,
                    self.data.size(),
                    $shader,
                    $entry_point,
                    dispatch_size,
                );

                let null_buffer = NullBitBufferGpu::clone_null_bit_buffer_pass(
                    &self.null_buffer,
                    &mut pipeline.encoder,
                );

                Self {
                    data: Arc::new(new_buffer),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer,
                }
            }
        }
    };
}

pub(crate) use impl_arithmetic_op;

macro_rules! impl_arithmetic_array_op {
    ($trait_name: ident, $array_type:ident, $trait_function: ident, $ty: ident, $shader: ident, $entry_point: expr) => {
        impl<T> $trait_name<PrimitiveArrayGpu<T>> for $ty
        where
            T: $array_type + ArrowPrimitiveType,
        {
            type Output = Self;

            fn $trait_function(
                &self,
                value: &PrimitiveArrayGpu<T>,
                pipeline: &mut ArrowComputePipeline,
            ) -> Self::Output {
                let dispatch_size = self.data.size().div_ceil(T::ITEM_SIZE).div_ceil(256) as u32;

                let new_data_buffer = pipeline.apply_binary_function(
                    &self.data,
                    &value.data,
                    self.data.size(),
                    $shader,
                    $entry_point,
                    dispatch_size,
                );

                let new_null_buffer = NullBitBufferGpu::merge_null_bit_buffer_op(
                    &self.null_buffer,
                    &value.null_buffer,
                    pipeline,
                );

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

#[cfg(test)]
mod test {
    macro_rules! test_sum {
        ($test_name: ident, $ty: ident, $base: expr, $size: expr, $output: expr) => {
            #[test]
            fn $test_name() {
                use arrow_gpu_array::{
                    gpu_utils::GpuDevice, kernels::broadcast::Broadcast, GPU_DEVICE,
                };
                let device = GPU_DEVICE
                    .get_or_init(|| Arc::new(GpuDevice::new()))
                    .clone();
                let array = $ty::broadcast($base, $size, device);
                assert_eq!(array.sum().raw_values().unwrap(), vec![$output]);
            }
        };
    }
    pub(crate) use test_sum;
}
