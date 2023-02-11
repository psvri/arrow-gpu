use bytemuck::Pod;
use std::{any::Any, fmt::Debug};
use wgpu::util::align_to;
pub mod date32_gpu;
pub mod f32_gpu;
pub(crate) mod gpu_ops;
pub mod i16_gpu;
pub mod i32_gpu;
pub mod i8_gpu;
pub mod primitive_array_gpu;
pub mod u16_gpu;
pub mod u32_gpu;
pub mod u8_gpu;

use std::{borrow::Cow, sync::Arc};

use log::info;
use wgpu::{util::DeviceExt, Adapter, Buffer, ComputePipeline, Device, Queue, ShaderModule};

use crate::array::gpu_ops::u32_ops::bit_and_array;

use date32_gpu::Date32ArrayGPU;
use date32_gpu::Date32Type;
use f32_gpu::Float32ArrayGPU;
use i16_gpu::Int16ArrayGPU;
use i32_gpu::Int32ArrayGPU;
use i8_gpu::Int8ArrayGPU;
use u16_gpu::UInt16ArrayGPU;
use u32_gpu::UInt32ArrayGPU;
use u8_gpu::UInt8ArrayGPU;

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

#[derive(Debug)]
pub struct GpuDevice {
    device: Device,
    queue: Queue,
}

impl GpuDevice {
    pub async fn new() -> GpuDevice {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        info!("{:?}", device);

        Self { device, queue }
    }

    pub async fn from_adapter(adapter: Adapter) -> GpuDevice {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }

    #[inline]
    pub fn create_shader_module(&self, shader: &str) -> ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
    }

    #[inline]
    pub fn create_compute_pipeline(&self, shader: &str, entry_point: &str) -> ComputePipeline {
        let cs_module = self.create_shader_module(shader);
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point,
            })
    }

    #[inline]
    pub fn create_gpu_buffer_with_data(&self, data: &[impl RustNativeType]) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Values Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    #[inline]
    pub fn create_empty_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_retrive_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_scalar_buffer(&self, value: &impl Pod) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalar Buffer"),
                contents: bytemuck::cast_slice(&[*value]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn clone_buffer(&self, buffer: &Buffer) -> Buffer {
        let staging_buffer = self.create_empty_buffer(buffer.size());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        let submission_index = self.queue.submit(Some(encoder.finish()));

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        staging_buffer
    }

    pub async fn retrive_data(&self, data: &Buffer) -> Vec<u8> {
        let size = data.size() as wgpu::BufferAddress;

        let staging_buffer = self.create_retrive_buffer(size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&data, 0, &staging_buffer, 0, size);

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = data.to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // /println!("{:?}", result);

            result
        } else {
            panic!("failed to run compute on gpu!");
        }
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
    fn new(gpu_device: Arc<GpuDevice>, buffer_builder: &NullBitBufferBuilder) -> Option<Self> {
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

    fn new_set_with_capacity(gpu_device: Arc<GpuDevice>, size: usize) -> Self {
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
                let new_bit_buffer =
                    bit_and_array(&left.gpu_device, &left.bit_buffer, &right.bit_buffer).await;
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
