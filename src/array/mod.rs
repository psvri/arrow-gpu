use bytemuck::Pod;
use std::fmt::Debug;
use wgpu::util::align_to;

pub mod gpu_array;

pub trait NativeType: Pod + Debug + Default {}

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
    data: Vec<u8>,
    len: usize,
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

        Self { data, len: size }
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
