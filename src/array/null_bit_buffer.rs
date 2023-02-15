use std::sync::Arc;

use wgpu::{util::align_to, Buffer};

use crate::array::u32_gpu::U32_ARRAY_SHADER;

use super::gpu_device::GpuDevice;

pub struct NullBitBufferBuilder {
    data: Vec<u8>,
    len: usize,
    contains_nulls: bool,
}

impl NullBitBufferBuilder {
    pub fn new() -> Self {
        Self::new_with_capacity(1024)
    }

    pub fn new_with_capacity(size: usize) -> Self {
        let aligned_size = align_to(size, 8) / 8;
        Self {
            data: vec![0; aligned_size],
            len: size,
            contains_nulls: true,
        }
    }

    pub fn new_set_with_capacity(size: usize) -> Self {
        let aligned_size = align_to(size, 8) / 8;
        let mut data = vec![u8::MAX; aligned_size];

        // set padding bits to zero
        let diff = size % 8;
        if diff != 0 {
            data[aligned_size - 1] = u8::MAX >> (8 - diff);
        }

        Self {
            data,
            len: size,
            contains_nulls: false,
        }
    }

    pub fn set_bit(&mut self, pos: usize) {
        self.data[pos / 8] |= 1 << (pos % 8);
    }

    pub fn unset_bit(&mut self, pos: usize) {
        self.data[pos / 8] &= !(1 << (pos % 8));
    }

    pub fn is_set(&self, pos: usize) -> bool {
        self.data[pos / 8] & 1 << (pos % 8) == 1 << (pos % 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_bit() {
        let mut buffer = NullBitBufferBuilder::new_with_capacity(10);
        assert_eq!(buffer.data.len(), 2);
        buffer.set_bit(0);
        assert_eq!(buffer.data[0], 0b00000001);
        buffer.set_bit(9);
        assert_eq!(buffer.data[1], 0b00000010);
        assert!(!buffer.is_set(5));
        assert!(buffer.is_set(9));
        assert!(buffer.is_set(0));
    }

    #[test]
    fn test_new_set_with_capacity() {
        let buffer = NullBitBufferBuilder::new_set_with_capacity(10);

        assert_eq!(buffer.data[0], u8::MAX);
        assert_eq!(buffer.data[1], 0b00000011);
    }
}

#[derive(Debug, Clone)]
pub struct NullBitBufferGpu {
    bit_buffer: Arc<Buffer>,
    len: usize,
    buffer_len: usize,
    gpu_device: Arc<GpuDevice>,
}

impl NullBitBufferGpu {
    pub fn new(gpu_device: Arc<GpuDevice>, buffer_builder: &NullBitBufferBuilder) -> Option<Self> {
        if buffer_builder.contains_nulls {
            let data = gpu_device.create_gpu_buffer_with_data(&buffer_builder.data);

            Some(Self {
                bit_buffer: Arc::new(data),
                len: buffer_builder.len,
                buffer_len: buffer_builder.data.len(),
                gpu_device,
            })
        } else {
            None
        }
    }

    pub fn new_set_with_capacity(gpu_device: Arc<GpuDevice>, size: usize) -> Self {
        let buffer_builder = NullBitBufferBuilder::new_set_with_capacity(size);
        let data = gpu_device.create_gpu_buffer_with_data(&buffer_builder.data);

        Self {
            bit_buffer: Arc::new(data),
            len: buffer_builder.len,
            buffer_len: buffer_builder.data.len(),
            gpu_device,
        }
    }

    pub async fn raw_values(&self) -> Vec<u8> {
        let result = &self.gpu_device.retrive_data(&self.bit_buffer).await;
        result[0..self.buffer_len].to_vec()
    }

    pub async fn merge_null_bit_buffer(
        left: &Option<NullBitBufferGpu>,
        right: &Option<NullBitBufferGpu>,
    ) -> Option<NullBitBufferGpu> {
        match (left, right) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x.clone()),
            (Some(left), Some(right)) => {
                assert_eq!(left.bit_buffer.size(), right.bit_buffer.size());
                assert_eq!(left.len, right.len);
                assert!(Arc::ptr_eq(&left.gpu_device, &right.gpu_device));
                let new_bit_buffer = left
                    .gpu_device
                    .apply_scalar_function(
                        &left.bit_buffer,
                        &right.bit_buffer,
                        left.bit_buffer.size(),
                        4,
                        U32_ARRAY_SHADER,
                        "bitwise_and",
                    )
                    .await;
                let len = left.len;
                let gpu_device = left.gpu_device.clone();

                Some(Self {
                    bit_buffer: Arc::new(new_bit_buffer),
                    len,
                    buffer_len: left.buffer_len,
                    gpu_device,
                })
            }
        }
    }
}
