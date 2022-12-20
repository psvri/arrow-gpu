pub mod f32_gpu;
pub(crate) mod gpu_ops;
pub mod primitive_array_gpu;
pub mod u32_gpu;
use pollster::FutureExt;

use std::{borrow::Cow, sync::Arc};

use bytemuck::Pod;
use log::info;
use wgpu::{util::DeviceExt, Adapter, Buffer, ComputePipeline, Device, Queue, ShaderModule};

use super::NullBitBufferBuilder;

#[derive(Debug, Clone)]
pub struct GpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl GpuDevice {
    pub async fn new() -> GpuDevice {
        let instance = wgpu::Instance::new(wgpu::Backends::all());

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

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
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

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
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
    pub fn create_gpu_buffer_with_data(&self, data: &[impl Pod]) -> Buffer {
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
}

#[derive(Debug, Clone)]
pub struct NullBitBufferGpu {
    bit_buffer: Arc<Buffer>,
    len: usize,
    gpu_device: GpuDevice,
}

impl NullBitBufferGpu {
    fn new(gpu_device: GpuDevice, buffer_builder: &NullBitBufferBuilder) -> Self {
        let data = gpu_device.create_gpu_buffer_with_data(&buffer_builder.data);

        Self {
            bit_buffer: Arc::new(data),
            len: buffer_builder.len,
            gpu_device,
        }
    }

    pub fn raw_values(&self) -> Option<Vec<u32>> {
        let size = self.bit_buffer.size();

        let staging_buffer = self.gpu_device.create_retrive_buffer(size);
        let mut encoder = self
            .gpu_device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.bit_buffer, 0, &staging_buffer, 0, size);

        let submission_index = self.gpu_device.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.gpu_device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().block_on() {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
            Some(result[0..(self.bit_buffer.size() / 4) as usize].to_vec())
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    async fn merge_null_bit_buffer(
        left: &Option<NullBitBufferGpu>,
        right: &Option<NullBitBufferGpu>,
    ) -> Option<NullBitBufferGpu> {
        match (left, right) {
            (None, None) => None,
            (None, Some(r)) | (Some(r), None) => {
                let bit_buffer = Arc::new(r.gpu_device.clone_buffer(&r.bit_buffer));

                Some(Self {
                    bit_buffer,
                    len: r.len,
                    gpu_device: r.gpu_device.clone(),
                })
            }
            (Some(l), Some(r)) => {
                assert_eq!(l.bit_buffer.size(), r.bit_buffer.size());
                let len = l.len;
                let gpu_device = l.gpu_device.clone();

                None
            }
        }
    }
}
