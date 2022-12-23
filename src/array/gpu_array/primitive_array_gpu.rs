use crate::array::{NativeType, NullBitBufferBuilder};

use pollster::FutureExt;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::align_to;
use wgpu::Buffer;

pub struct PrimitiveArrayGpu<T: NativeType> {
    pub(crate) data: Arc<Buffer>,
    pub(crate) gpu_device: Arc<GpuDevice>,
    pub(crate) phantom: PhantomData<T>,
    /// Actual len of the array
    pub(crate) len: usize,
    pub(crate) null_buffer: NullBitBufferGpu,
}

impl<T: NativeType> PrimitiveArrayGpu<T> {
    pub fn from_optional_slice(value: &[Option<T>], gpu_device: Arc<GpuDevice>) -> Self {
        let element_size = std::mem::size_of::<T>();

        let aligned_size = align_to(value.len() * element_size, 4);
        let mut new_vec = Vec::<T>::with_capacity(aligned_size / element_size);
        let mut null_buffer_builder = NullBitBufferBuilder::new_with_capacity(value.len());

        for (index, val) in value.iter().enumerate() {
            match val {
                Some(x) => {
                    new_vec.push(*x);
                    null_buffer_builder.set_bit(index);
                }
                None => {
                    new_vec.push(T::default());
                }
            }
        }

        let data = gpu_device.create_gpu_buffer_with_data(&new_vec);
        let null_buffer = NullBitBufferGpu::new(gpu_device.clone(), &null_buffer_builder);

        Self {
            data: Arc::new(data),
            gpu_device,
            phantom: Default::default(),
            len: value.len(),
            null_buffer,
        }
    }

    pub fn from_optional_vec(value: &Vec<Option<T>>, gpu_device: Arc<GpuDevice>) -> Self {
        Self::from_optional_slice(&value[..], gpu_device)
    }

    pub fn from_slice(value: &[T], gpu_device: Arc<GpuDevice>) -> Self {
        let data = gpu_device.create_gpu_buffer_with_data(value);
        let null_buffer = NullBitBufferGpu::new_set_with_capacity(gpu_device.clone(), value.len());

        Self {
            data: Arc::new(data),
            gpu_device,
            phantom: Default::default(),
            len: value.len(),
            null_buffer,
        }
    }

    pub fn from_vec(value: &Vec<T>, gpu_device: Arc<GpuDevice>) -> Self {
        Self::from_slice(&value[..], gpu_device)
    }

    pub fn raw_values(&self) -> Option<Vec<T>> {
        let size = self.data.size() as wgpu::BufferAddress;

        let staging_buffer = self.gpu_device.create_retrive_buffer(size);
        let mut encoder = self
            .gpu_device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.data, 0, &staging_buffer, 0, size);

        let submission_index = self.gpu_device.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.gpu_device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().block_on() {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
            Some(result[0..self.len].to_vec())
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    pub fn values(&self) -> Vec<Option<T>> {
        match self.raw_values() {
            Some(primitive_values) => {
                let mut result_vec = Vec::with_capacity(self.len);

                // TODO rework this
                match self.null_buffer.raw_values() {
                    Some(null_bit_buffer) => {
                        for (pos, val) in primitive_values.iter().enumerate() {
                            if (null_bit_buffer[pos / 8] & 1 << (pos % 8)) == 1 << (pos % 8) {
                                result_vec.push(Some(*val))
                            } else {
                                result_vec.push(None)
                            }
                        }
                    }
                    None => {
                        for val in primitive_values {
                            result_vec.push(Some(val))
                        }
                    }
                }

                result_vec
            }
            None => vec![],
        }
    }
}

impl<T: NativeType> Debug for PrimitiveArrayGpu<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "{:?}", self.data)?;
        writeln!(f, "{:?}", self.gpu_device.device)?;
        writeln!(f, "{:?}", self.gpu_device.queue)?;
        writeln!(
            f,
            "Array of length {} contains {:?}",
            self.len,
            self.values()
        )?;
        write!(f, "}}")
    }
}

macro_rules! impl_scalar_ops {
    ($trait_name: ident, $trait_function: ident, $ty: ident, $op: ident) => {
        #[async_trait]
        impl $trait_name<$ty> for PrimitiveArrayGpu<$ty> {
            type Output = Self;

            async fn $trait_function(&self, value: &$ty) -> Self::Output {
                let new_buffer = $op(&self.gpu_device, &self.data, *value).await;

                Self {
                    data: Arc::new(new_buffer),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer: self.null_buffer.clone(),
                }
            }
        }
    };
}

pub(crate) use impl_scalar_ops;

macro_rules! impl_add_trait {
    ($ty: ident, $op: ident) => {
        impl_scalar_ops!(ArrowAdd, add, $ty, $op);
    };
}

pub(crate) use impl_add_trait;

macro_rules! impl_sub_trait {
    ($ty: ident, $op: ident) => {
        impl_scalar_ops!(ArrowSub, sub, $ty, $op);
    };
}

pub(crate) use impl_sub_trait;

macro_rules! impl_mul_trait {
    ($ty: ident, $op: ident) => {
        impl_scalar_ops!(ArrowMul, mul, $ty, $op);
    };
}

pub(crate) use impl_mul_trait;

macro_rules! impl_div_trait {
    ($ty: ident, $op: ident) => {
        impl_scalar_ops!(ArrowDiv, div, $ty, $op);
    };
}

pub(crate) use impl_div_trait;

macro_rules! impl_array_ops {
    ($trait_name: ident, $trait_function: ident, $for_ty: ident, $rhs_ty: ident, $op: ident) => {
        #[async_trait]
        impl $trait_name<$rhs_ty> for $for_ty {
            type Output = Self;

            async fn $trait_function(&self, value: &$rhs_ty) -> Self::Output {
                assert!(Arc::ptr_eq(&self.gpu_device, &value.gpu_device));
                let new_data_buffer = $op(&self.gpu_device, &self.data, &value.data).await;
                let new_null_buffer =
                    NullBitBufferGpu::merge_null_bit_buffer(&self.null_buffer, &value.null_buffer)
                        .await;

                let result = Self {
                    data: Arc::new(new_data_buffer),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer: new_null_buffer,
                };

                result
            }
        }
    };
}

pub(crate) use impl_array_ops;

macro_rules! impl_array_add_trait {
    ($for_ty: ident, $rhs_ty: ident, $op: ident) => {
        impl_array_ops!(ArrowAdd, add, $for_ty, $rhs_ty, $op);
    };
}

pub(crate) use impl_array_add_trait;

macro_rules! impl_add_assign_trait {
    ($ty: ident, $op: ident) => {
        #[async_trait]
        impl ArrowAddAssign<$ty> for PrimitiveArrayGpu<$ty> {
            async fn add_assign(&mut self, value: &$ty) {
                $op(&self.gpu_device, &self.data, *value).await;
            }
        }
    };
}

pub(crate) use impl_add_assign_trait;

use super::{GpuDevice, NullBitBufferGpu};

#[cfg(test)]
pub mod test {
    macro_rules! test_scalar_op {
        ($fn_name: ident, $ty: ident, $input: expr, $scalar_fn: ident, $scalar: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
                let data = $input;
                let gpu_array = PrimitiveArrayGpu::<$ty>::from_vec(&data, device);
                let new_gpu_array = gpu_array.$scalar_fn($scalar).await;
                assert_eq!(gpu_array.raw_values().unwrap(), data);
                assert_eq!(new_gpu_array.raw_values().unwrap(), $output);
            }
        };
    }
    pub(crate) use test_scalar_op;

    macro_rules! test_add_array {
        ($fn_name: ident, $ty: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
                let gpu_array_1 = $ty::from_optional_vec(&$input_1, device.clone());
                let gpu_array_2 = $ty::from_optional_vec(&$input_2, device);
                let new_gpu_array = gpu_array_1.add(&gpu_array_2).await;
                assert_eq!(new_gpu_array.values(), $output);
            }
        };
    }
    pub(crate) use test_add_array;

    macro_rules! test_add_assign_scalar {
        ($fn_name: ident, $ty: ident, $input: expr, $scalar: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
                let data = $input;
                let mut gpu_array = PrimitiveArrayGpu::<$ty>::from_vec(&data, device);
                gpu_array.add_assign($scalar).await;
                assert_eq!(gpu_array.raw_values().unwrap(), $output)
            }
        };
        ($fn_name: ident, $ty: ident, $input: expr, $scalar: expr, $output_raw: expr, $output_values:expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::gpu_array::GpuDevice::new().await);
                let data = $input;
                let mut gpu_array = PrimitiveArrayGpu::<$ty>::from_optional_vec(&data, device);
                gpu_array.add_assign($scalar).await;
                assert_eq!(gpu_array.raw_values().unwrap(), $output_raw);
                assert_eq!(gpu_array.values(), $output_values);
            }
        };
    }
    pub(crate) use test_add_assign_scalar;
}
