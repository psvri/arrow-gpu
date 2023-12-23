use super::{gpu_device::GpuDevice, NullBitBufferGpu};
use crate::array::{ArrowPrimitiveType, BooleanBufferBuilder};

use pollster::FutureExt;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::align_to;
use wgpu::Buffer;

pub struct PrimitiveArrayGpu<T: ArrowPrimitiveType> {
    pub data: Arc<Buffer>,
    pub gpu_device: Arc<GpuDevice>,
    pub phantom: PhantomData<T>,
    /// Actual len of the array
    pub len: usize,
    pub null_buffer: Option<NullBitBufferGpu>,
}

impl<T: ArrowPrimitiveType> PrimitiveArrayGpu<T> {
    pub fn from_optional_slice(
        value: &[Option<T::NativeType>],
        gpu_device: Arc<GpuDevice>,
    ) -> Self {
        let element_size = T::ITEM_SIZE;

        let aligned_size = align_to(value.len() as u64 * element_size, 4);
        let mut new_vec =
            Vec::<T::NativeType>::with_capacity((aligned_size / element_size) as usize);
        let mut null_buffer_builder = BooleanBufferBuilder::new_with_capacity(value.len());

        for (index, val) in value.iter().enumerate() {
            match val {
                Some(x) => {
                    new_vec.push(*x);
                    null_buffer_builder.set_bit(index);
                }
                None => {
                    new_vec.push(T::NativeType::default());
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

    pub fn from_slice(value: &[T::NativeType], gpu_device: Arc<GpuDevice>) -> Self {
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

    pub async fn raw_values(&self) -> Option<Vec<T::NativeType>> {
        let result = self.gpu_device.retrive_data(&self.data).await;
        let result: Vec<T::NativeType> = bytemuck::cast_slice(&result).to_vec();
        Some(result[0..self.len].to_vec())
    }

    pub async fn values(&self) -> Vec<Option<T::NativeType>> {
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

    pub async fn clone_array(&self) -> Self {
        let data = self.gpu_device.clone_buffer(&self.data);
        let null_buffer = NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer);
        Self {
            data: Arc::new(data.await),
            gpu_device: self.gpu_device.clone(),
            phantom: PhantomData,
            len: self.len,
            null_buffer: null_buffer.await,
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

#[cfg(test)]
pub mod test {
    macro_rules! test_broadcast {
        ($fn_name: ident, $ty: ident, $input: expr) => {
            #[tokio::test]
            async fn $fn_name() {
                use crate::GPU_DEVICE;
                use crate::GpuDevice;
                use pollster::FutureExt;
                let device = GPU_DEVICE.get_or_init(|| std::sync::Arc::new(GpuDevice::new().block_on()).clone());
                let length = 100;
                let new_gpu_array = $ty::broadcast($input, length, device.clone()).await;
                let new_values = new_gpu_array.raw_values().await.unwrap();
                assert_eq!(new_values, vec![$input; length.try_into().unwrap()]);
            }
        };
    }
    pub(crate) use test_broadcast;
}
