pub mod f32_gpu;
pub mod primitive_array_gpu;

use std::{borrow::Cow, sync::Arc};

use log::{error, info};
use pollster::FutureExt;
use wgpu::{Adapter, ComputePipeline, Device, Queue, ShaderModule};

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

    pub fn create_shader_module(&self, shader: &str) -> ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
    }

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
}
