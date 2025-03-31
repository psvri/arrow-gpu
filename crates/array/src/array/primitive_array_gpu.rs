use super::ArrayUtils;
use super::NullBitBufferGpu;
use crate::array::{ArrowPrimitiveType, BooleanBufferBuilder};
use crate::gpu_utils::*;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::Buffer;
use wgpu::util::align_to;

/// Arrow array backed by primitive types stored in GPU
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

    pub fn raw_values(&self) -> Option<Vec<T::NativeType>> {
        let result = self.gpu_device.retrive_data(&self.data);
        let result: Vec<T::NativeType> = bytemuck::cast_slice(&result).to_vec();
        Some(result[0..self.len].to_vec())
    }

    pub fn values(&self) -> Vec<Option<T::NativeType>> {
        match self.raw_values() {
            Some(primitive_values) => {
                let mut result_vec = Vec::with_capacity(self.len);

                // TODO rework this
                match &self.null_buffer {
                    Some(null_bit_buffer) => {
                        let null_values = null_bit_buffer.raw_values();
                        for (pos, val) in primitive_values.iter().enumerate() {
                            if (null_values[pos / 8] & (1 << (pos % 8))) == 1 << (pos % 8) {
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

    pub fn clone_array(&self) -> Self {
        let data = self.gpu_device.clone_buffer(&self.data);
        let null_buffer = NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer);
        Self {
            data: Arc::new(data),
            gpu_device: self.gpu_device.clone(),
            phantom: PhantomData,
            len: self.len,
            null_buffer,
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
            self.values()
        )?;
        write!(f, "}}")
    }
}

impl<T: ArrowPrimitiveType> ArrayUtils for PrimitiveArrayGpu<T> {
    fn get_gpu_device(&self) -> Arc<GpuDevice> {
        self.gpu_device.clone()
    }
}

#[cfg(test)]
pub mod test {
    macro_rules! test_broadcast {
        ($fn_name: ident, $ty: ident, $input: expr) => {
            #[test]
            fn $fn_name() {
                use crate::GPU_DEVICE;
                let device = GPU_DEVICE.clone();
                let length = 100;
                let new_gpu_array = $ty::broadcast($input, length, device.clone());
                let new_values = new_gpu_array.raw_values().unwrap();
                assert_eq!(new_values, vec![$input; length.try_into().unwrap()]);
            }
        };
    }
    pub(crate) use test_broadcast;
}
