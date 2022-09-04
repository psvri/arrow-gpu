pub mod primitive_array_gpu;

use pollster::FutureExt;
use wgpu::{Device, Queue};

fn get_device() -> (Device, Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .block_on()
        .unwrap();

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .block_on()
        .unwrap()
}
