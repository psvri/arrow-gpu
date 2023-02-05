use super::{GpuDevice, NullBitBufferGpu};
use crate::array::{ArrowPrimitiveType, NullBitBufferBuilder};

use pollster::FutureExt;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::align_to;
use wgpu::Buffer;

pub struct PrimitiveArrayGpu<T: ArrowPrimitiveType> {
    pub(crate) data: Arc<Buffer>,
    pub(crate) gpu_device: Arc<GpuDevice>,
    pub(crate) phantom: PhantomData<T>,
    /// Actual len of the array
    pub(crate) len: usize,
    pub(crate) null_buffer: Option<NullBitBufferGpu>,
}

impl<T: ArrowPrimitiveType> PrimitiveArrayGpu<T> {
    pub fn from_optional_slice(value: &[Option<T>], gpu_device: Arc<GpuDevice>) -> Self {
        let element_size = T::ITEM_SIZE;

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
        let null_buffer = None;

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

    pub async fn raw_values(&self) -> Option<Vec<T>> {
        let result = &self.gpu_device.retrive_data(&self.data).await;
        let result: Vec<T> = bytemuck::cast_slice(&result).to_vec();
        Some(result[0..self.len].to_vec())
    }

    pub async fn values(&self) -> Vec<Option<T>> {
        match self.raw_values().await {
            Some(primitive_values) => {
                let mut result_vec = Vec::with_capacity(self.len);

                // TODO rework this
                match &self.null_buffer {
                    Some(null_bit_buffer) => {
                        let null_values = null_bit_buffer.raw_values().await;
                        for (pos, val) in primitive_values.iter().enumerate() {
                            if (null_values[pos / 8] & 1 << (pos % 8)) == 1 << (pos % 8) {
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

impl<T: ArrowPrimitiveType> Debug for PrimitiveArrayGpu<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "{:?}", self.data)?;
        writeln!(f, "{:?}", self.gpu_device.device)?;
        writeln!(f, "{:?}", self.gpu_device.queue)?;
        writeln!(
            f,
            "Array of length {} contains {:?}",
            self.len,
            self.values().block_on()
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

macro_rules! impl_unary_ops {
    ($trait_name: ident, $trait_function: ident, $for_ty: ident, $out_ty: ident, $op: ident) => {
        #[async_trait]
        impl $trait_name for $for_ty {
            type Output = $out_ty;

            async fn $trait_function(&self) -> Self::Output {
                $op(&self.gpu_device, &self.data, self.len).await
            }
        }
    };
}

pub(crate) use impl_unary_ops;

macro_rules! impl_array_add_trait {
    ($for_ty: ident, $rhs_ty: ident, $op: ident) => {
        impl_array_ops!(ArrowAdd, add, $for_ty, $rhs_ty, $op);
    };
}

#[cfg(test)]
pub mod test {
    macro_rules! test_scalar_op {
        ($fn_name: ident, $ty: ident, $input: expr, $scalar_fn: ident, $scalar: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::GpuDevice::new().await);
                let data = $input;
                let gpu_array = PrimitiveArrayGpu::<$ty>::from_vec(&data, device);
                let new_gpu_array = gpu_array.$scalar_fn($scalar).await;
                assert_eq!(gpu_array.raw_values().await.unwrap(), data);
                assert_eq!(new_gpu_array.raw_values().await.unwrap(), $output);
            }
        };
    }
    pub(crate) use test_scalar_op;

    macro_rules! test_add_array {
        ($fn_name: ident, $ty: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::GpuDevice::new().await);
                let gpu_array_1 = $ty::from_optional_vec(&$input_1, device.clone());
                let gpu_array_2 = $ty::from_optional_vec(&$input_2, device);
                let new_gpu_array = gpu_array_1.add(&gpu_array_2).await;
                assert_eq!(new_gpu_array.values().await, $output);
            }
        };
    }
    pub(crate) use test_add_array;

    pub fn float_eq_in_error(left: f32, right: f32) -> bool {
        if (left == f32::INFINITY && right != f32::INFINITY)
            || (right == f32::INFINITY && left != f32::INFINITY)
            || (right == f32::NEG_INFINITY && left != f32::NEG_INFINITY)
            || (left == f32::NEG_INFINITY && right != f32::NEG_INFINITY)
            || (left.is_nan() && !right.is_nan())
            || (right.is_nan() && !left.is_nan())
            || (left - right) > 0.0001
        {
            false
        } else {
            true
        }
    }

    macro_rules! test_unary_op_float {
        ($fn_name: ident, $ty: ident, $input: expr, $unary_fn: ident, $unary_fn_dyn: ident, $output: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::GpuDevice::new().await);
                let data = $input;
                let gpu_array = $ty::from_vec(&data, device);
                let new_gpu_array = gpu_array.$unary_fn().await;
                let new_values = new_gpu_array.raw_values().await.unwrap();
                for (index, new_value) in new_values.iter().enumerate() {
                    if !float_eq_in_error($output[index], *new_value) {
                        panic!(
                            "assertion failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                            $output, new_values
                        );
                    }
                }

                let new_gpu_array = $unary_fn_dyn(&gpu_array).await;
                let new_values = (*new_gpu_array.downcast_ref::<$ty>().unwrap())
                    .raw_values()
                    .await
                    .unwrap();
                for (index, new_value) in new_values.iter().enumerate() {
                    if !float_eq_in_error($output[index], *new_value) {
                        panic!(
                            "assertion dyn failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                            $output, new_values
                        );
                    }
                }
            }
        };
    }
    pub(crate) use test_unary_op_float;

    macro_rules! test_broadcast {
        ($fn_name: ident, $ty: ident, $input: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                let device = Arc::new(crate::array::GpuDevice::new().await);
                let length = 100;
                let new_gpu_array =
                    PrimitiveArrayGpu::<$ty>::braodcast($input, length, device).await;
                let new_values = new_gpu_array.raw_values().await.unwrap();
                assert_eq!(new_values, vec![$input; length]);
            }
        };
    }
    pub(crate) use test_broadcast;
}
