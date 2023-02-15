use bytemuck::Pod;
use std::{any::Any, fmt::Debug};
pub mod date32_gpu;
pub mod f32_gpu;
pub mod gpu_device;
pub(crate) mod gpu_ops;
pub mod i16_gpu;
pub mod i32_gpu;
pub mod i8_gpu;
pub mod null_bit_buffer;
pub mod primitive_array_gpu;
pub mod u16_gpu;
pub mod u32_gpu;
pub mod u8_gpu;

use null_bit_buffer::*;
use wgpu::Buffer;

use date32_gpu::Date32ArrayGPU;
use date32_gpu::Date32Type;
use f32_gpu::Float32ArrayGPU;
use i16_gpu::Int16ArrayGPU;
use i32_gpu::Int32ArrayGPU;
use i8_gpu::Int8ArrayGPU;
use u16_gpu::UInt16ArrayGPU;
use u32_gpu::UInt32ArrayGPU;
use u8_gpu::UInt8ArrayGPU;

use self::gpu_device::GpuDevice;

#[derive(Debug)]
#[non_exhaustive]
pub enum ArrowType {
    Float32Type,
    UInt32Type,
    UInt16Type,
    UInt8Type,
    Int32Type,
    Int16Type,
    Int8Type,
}

pub trait ArrowPrimitiveType: Send + Sync {
    type NativeType: RustNativeType;
    const ITEM_SIZE: usize;
}

pub trait RustNativeType: Pod + Debug + Default {}

impl RustNativeType for i32 {}
impl RustNativeType for i16 {}
impl RustNativeType for i8 {}
impl RustNativeType for f32 {}
impl RustNativeType for u32 {}
impl RustNativeType for u16 {}
impl RustNativeType for u8 {}

macro_rules! impl_primitive_type {
    ($primitive_type: ident, $t: ident, $size: expr) => {
        impl ArrowPrimitiveType for $primitive_type {
            type NativeType = $t;
            const ITEM_SIZE: usize = $size;
        }
    };
}

impl_primitive_type!(f32, f32, 4);
impl_primitive_type!(u32, u32, 4);
impl_primitive_type!(u16, u16, 2);
impl_primitive_type!(u8, u8, 1);
impl_primitive_type!(i32, i32, 4);
impl_primitive_type!(i16, i16, 4);
impl_primitive_type!(i8, i8, 1);
impl_primitive_type!(Date32Type, i32, 4);

pub trait ArrowArray: Any + Sync + Send + Debug {
    fn as_any(&self) -> &dyn Any;
    fn get_data_type(&self) -> ArrowType;
    fn get_memory_used(&self) -> u64;
    fn get_gpu_device(&self) -> &GpuDevice;
    fn get_buffer(&self) -> &Buffer;
    fn get_null_bit_buffer(&self) -> Option<&NullBitBufferGpu>;
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ArrowArrayGPU {
    Float32ArrayGPU(Float32ArrayGPU),
    UInt32ArrayGPU(UInt32ArrayGPU),
    UInt16ArrayGPU(UInt16ArrayGPU),
    UInt8ArrayGPU(UInt8ArrayGPU),
    Int32ArrayGPU(Int32ArrayGPU),
    Int16ArrayGPU(Int16ArrayGPU),
    Int8ArrayGPU(Int8ArrayGPU),
    Date32ArrayGPU(Date32ArrayGPU),
}
