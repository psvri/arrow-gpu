use std::sync::Arc;

use wgpu::{util::align_to, Buffer, CommandEncoder};

const LOGICAL_AND_SHADER: &str = include_str!("../../../logical/compute_shaders/u32/logical.wgsl");

use super::{gpu_device::GpuDevice, ArrowComputePipeline};

pub struct BooleanBufferBuilder {
    pub(crate) data: Vec<u8>,
    len: usize,
    contains_nulls: bool,
}

impl BooleanBufferBuilder {
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

    pub fn is_set_in_slice(data: &[u8], pos: usize) -> bool {
        data[pos / 8] & 1 << (pos % 8) == 1 << (pos % 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_bit() {
        let mut buffer = BooleanBufferBuilder::new_with_capacity(10);
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
        let buffer = BooleanBufferBuilder::new_set_with_capacity(10);

        assert_eq!(buffer.data[0], u8::MAX);
        assert_eq!(buffer.data[1], 0b00000011);
    }
}

#[derive(Debug, Clone)]
pub struct NullBitBufferGpu {
    pub bit_buffer: Arc<Buffer>,
    pub len: usize,
    pub gpu_device: Arc<GpuDevice>,
}

impl NullBitBufferGpu {
    pub fn new(gpu_device: Arc<GpuDevice>, buffer_builder: &BooleanBufferBuilder) -> Option<Self> {
        if buffer_builder.contains_nulls {
            let data = gpu_device.create_gpu_buffer_with_data(&buffer_builder.data);

            Some(Self {
                bit_buffer: Arc::new(data),
                len: buffer_builder.len,
                gpu_device,
            })
        } else {
            None
        }
    }

    pub fn new_set_with_capacity(gpu_device: Arc<GpuDevice>, size: usize) -> Self {
        let buffer_builder = BooleanBufferBuilder::new_set_with_capacity(size);
        let data = gpu_device.create_gpu_buffer_with_data(&buffer_builder.data);

        Self {
            bit_buffer: Arc::new(data),
            len: buffer_builder.len,
            gpu_device,
        }
    }

    pub fn raw_values(&self) -> Vec<u8> {
        let result = &self.gpu_device.retrive_data(&self.bit_buffer);
        let buffer_size = align_to(self.len, 8) / 8;
        result[0..buffer_size].to_vec()
    }

    pub fn clone_null_bit_buffer(data: &Option<Self>) -> Option<Self> {
        data.as_ref().map(|null_bit_buffer| NullBitBufferGpu {
            bit_buffer: Arc::new(null_bit_buffer.clone_buffer()),
            len: null_bit_buffer.len,
            gpu_device: null_bit_buffer.gpu_device.clone(),
        })
    }

    pub fn clone_null_bit_buffer_pass(
        data: &Option<Self>,
        encoder: &mut CommandEncoder,
    ) -> Option<Self> {
        data.as_ref().map(|null_bit_buffer| NullBitBufferGpu {
            bit_buffer: Arc::new(null_bit_buffer.clone_buffer_pass(encoder)),
            len: null_bit_buffer.len,
            gpu_device: null_bit_buffer.gpu_device.clone(),
        })
    }

    fn clone_buffer(&self) -> Buffer {
        self.gpu_device.clone_buffer(&self.bit_buffer)
    }

    fn clone_buffer_pass(&self, encoder: &mut CommandEncoder) -> Buffer {
        self.gpu_device.clone_buffer_pass(&self.bit_buffer, encoder)
    }

    pub fn merge_null_bit_buffer(
        left: &Option<NullBitBufferGpu>,
        right: &Option<NullBitBufferGpu>,
    ) -> Option<NullBitBufferGpu> {
        match (left, right) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some({
                let buffer = x.clone_buffer();
                Self {
                    bit_buffer: buffer.into(),
                    len: x.len,
                    gpu_device: x.gpu_device.clone(),
                }
            }),
            (Some(left), Some(right)) => {
                assert_eq!(left.bit_buffer.size(), right.bit_buffer.size());
                assert_eq!(left.len, right.len);
                assert!(Arc::ptr_eq(&left.gpu_device, &right.gpu_device));
                let new_bit_buffer = left.gpu_device.apply_scalar_function(
                    &left.bit_buffer,
                    &right.bit_buffer,
                    left.bit_buffer.size(),
                    4,
                    LOGICAL_AND_SHADER,
                    "bitwise_and",
                );
                let len = left.len;
                let gpu_device = left.gpu_device.clone();

                Some(Self {
                    bit_buffer: Arc::new(new_bit_buffer),
                    len,
                    gpu_device,
                })
            }
        }
    }

    pub fn merge_null_bit_buffer_op(
        left: &Option<NullBitBufferGpu>,
        right: &Option<NullBitBufferGpu>,
        pipeline: &mut ArrowComputePipeline,
    ) -> Option<NullBitBufferGpu> {
        match (left, right) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some({
                let buffer = x.clone_buffer_pass(&mut pipeline.encoder);
                Self {
                    bit_buffer: buffer.into(),
                    len: x.len,
                    gpu_device: x.gpu_device.clone(),
                }
            }),
            (Some(left), Some(right)) => {
                assert_eq!(left.bit_buffer.size(), right.bit_buffer.size());
                assert_eq!(left.len, right.len);
                assert!(Arc::ptr_eq(&left.gpu_device, &right.gpu_device));
                let new_bit_buffer = left.gpu_device.apply_scalar_function(
                    &left.bit_buffer,
                    &right.bit_buffer,
                    left.bit_buffer.size(),
                    4,
                    LOGICAL_AND_SHADER,
                    "bitwise_and",
                );
                let len = left.len;
                let gpu_device = left.gpu_device.clone();

                Some(Self {
                    bit_buffer: Arc::new(new_bit_buffer),
                    len,
                    gpu_device,
                })
            }
        }
    }
}
