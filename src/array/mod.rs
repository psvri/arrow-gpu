use bytemuck::Pod;
use std::fmt::Debug;
use wgpu::util::align_to;

pub mod gpu_array;

pub trait NativeType: Pod + Debug {}

impl NativeType for f32 {}
impl NativeType for i64 {}
impl NativeType for i32 {}
impl NativeType for i16 {}
impl NativeType for i8 {}
impl NativeType for u64 {}
impl NativeType for u32 {}
impl NativeType for u16 {}
impl NativeType for u8 {}

pub struct NullBitBufferBuilder {
    data: Vec<u32>,
    len: usize,
}

impl NullBitBufferBuilder {
    pub fn new() -> Self {
        Self::new_with_capacity(1024)
    }

    pub fn new_with_capacity(size: usize) -> Self {
        let aligned_size = align_to((size as f64 / 32.0).ceil() as usize, 4);
        Self {
            data: vec![0; aligned_size],
            len: size,
        }
    }

    pub fn set_bit(&mut self, pos: usize) {
        let index = pos / 32;
        let bit_index = pos % 32;
        self.data[index] |= 1 << bit_index;
    }

    pub fn is_set(&self, pos: usize) -> bool {
        let index = pos / 32;
        let bit_index = pos % 32;
        self.data[index] & 1 << bit_index == 1 << bit_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_bit() {
        let mut buffer = NullBitBufferBuilder::new_with_capacity(10);
        assert_eq!(buffer.data.len(), 4);
        buffer.set_bit(0);
        assert_eq!(buffer.data[0], 0b00000000_00000000_00000000_00000001);
        buffer.set_bit(9);
        assert_eq!(buffer.data[0], 0b00000000_00000000_00000010_00000001);
        assert!(!buffer.is_set(5));
        assert!(buffer.is_set(9));
        assert!(buffer.is_set(0));
    }
}
