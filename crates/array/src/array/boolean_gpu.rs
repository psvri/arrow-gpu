use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use async_trait::async_trait;
use pollster::FutureExt;
use wgpu::Buffer;

use crate::kernels::broadcast::Broadcast;
use crate::ArrowErrorGPU;

use super::{ArrowArrayGPU, BooleanBufferBuilder, GpuDevice, NullBitBufferGpu};

pub struct BooleanArrayGPU {
    pub data: Arc<Buffer>,
    pub gpu_device: Arc<GpuDevice>,
    /// Actual len of the array
    pub len: usize,
    pub null_buffer: Option<NullBitBufferGpu>,
}

impl BooleanArrayGPU {
    pub fn from_optional_slice(value: &[Option<bool>], gpu_device: Arc<GpuDevice>) -> Self {
        let mut buffer = BooleanBufferBuilder::new_with_capacity(value.len());
        let mut null_buffer_builder = BooleanBufferBuilder::new_with_capacity(value.len());

        for (index, val) in value.iter().enumerate() {
            match val {
                Some(true) => {
                    buffer.set_bit(index);
                    null_buffer_builder.set_bit(index);
                }
                Some(false) => {
                    null_buffer_builder.set_bit(index);
                }
                _ => {}
            }
        }

        let data = gpu_device.create_gpu_buffer_with_data(&buffer.data);
        let null_buffer = NullBitBufferGpu::new(gpu_device.clone(), &null_buffer_builder);

        Self {
            data: Arc::new(data),
            gpu_device,
            len: value.len(),
            null_buffer,
        }
    }

    pub fn from_optional_vec(value: &Vec<Option<bool>>, gpu_device: Arc<GpuDevice>) -> Self {
        Self::from_optional_slice(&value[..], gpu_device)
    }

    pub fn from_bytes_slice(value: &[u8], gpu_device: Arc<GpuDevice>) -> Self {
        let data = gpu_device.create_gpu_buffer_with_data(value);
        let null_buffer = None;

        Self {
            data: Arc::new(data),
            gpu_device,
            len: value.len(),
            null_buffer,
        }
    }

    pub fn from_bytes_vec(value: &Vec<u8>, gpu_device: Arc<GpuDevice>) -> Self {
        Self::from_bytes_slice(&value[..], gpu_device)
    }

    pub async fn raw_values(&self) -> Option<Vec<bool>> {
        let result = self.gpu_device.retrive_data(&self.data).await;
        let result: Vec<u8> = bytemuck::cast_slice(&result).to_vec();
        let mut bool_result = Vec::<bool>::with_capacity(self.len);
        for i in 0..self.len {
            bool_result.push(BooleanBufferBuilder::is_set_in_slice(&result, i))
        }
        Some(bool_result)
    }

    pub async fn values(&self) -> Vec<Option<bool>> {
        let primitive_values = self.raw_values().await.unwrap();
        let mut result_vec = Vec::with_capacity(self.len);

        // TODO rework this
        match &self.null_buffer {
            Some(null_bit_buffer) => {
                let null_values = null_bit_buffer.raw_values().await;
                for (pos, val) in primitive_values.iter().enumerate() {
                    if (null_values[pos / 8] & 1 << (pos % 8)) != 0 {
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
}

impl Debug for BooleanArrayGPU {
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

impl From<BooleanArrayGPU> for ArrowArrayGPU {
    fn from(val: BooleanArrayGPU) -> Self {
        ArrowArrayGPU::BooleanArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for BooleanArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::BooleanArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Date32ArrayGPU",
                x
            ))),
        }
    }
}

#[async_trait]
impl Broadcast<bool> for BooleanArrayGPU {
    type Output = BooleanArrayGPU;

    async fn broadcast(value: bool, len: usize, gpu_device: Arc<GpuDevice>) -> Self::Output {
        let null_buffer_builder = BooleanBufferBuilder::new_set_with_capacity(len);

        let buffer = if value {
            BooleanBufferBuilder::new_set_with_capacity(len)
        } else {
            BooleanBufferBuilder::new_with_capacity(len)
        };

        let data = gpu_device.create_gpu_buffer_with_data(&buffer.data);
        let null_buffer = NullBitBufferGpu::new(gpu_device.clone(), &null_buffer_builder);

        Self {
            data: Arc::new(data),
            gpu_device,
            len,
            null_buffer,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::primitive_array_gpu::test::test_broadcast;

    use super::*;

    #[tokio::test]
    async fn test_boolean_values() {
        let gpu_device = GpuDevice::new().await;
        let values = vec![Some(true), Some(true), Some(false), None];
        let array = BooleanArrayGPU::from_optional_slice(&values, Arc::new(gpu_device));

        let raw_values = array.raw_values().await.unwrap();
        assert_eq!(raw_values, vec![true, true, false, false]);

        let gpu_values = array.values().await;
        assert_eq!(gpu_values, values);
    }

    test_broadcast!(test_broadcast_bool, BooleanArrayGPU, true);
}
