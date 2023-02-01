use std::sync::Arc;

use arrow_gpu::array::f32_gpu::Float32ArrayGPU;
use arrow_gpu::array::u32_gpu::UInt32ArrayGPU;
use arrow_gpu::array::GpuDevice;
use arrow_gpu::kernels::arithmetic::*;
use pollster::FutureExt;

#[tokio::test]
async fn main() {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    instance
        .enumerate_adapters(wgpu::Backends::all())
        .for_each(|x| {
            println!("{:?}", x.get_info());
            println!("{:?}", x.features());
            println!("{:?}", x.limits());
        });

    println!("High power is {:?}", instance);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    println!("{:?}", adapter.get_info());
    println!("{:?}", adapter.features());
    println!("{:?}", adapter.limits());

    env_logger::init();

    let device = Arc::new(GpuDevice::new().block_on());

    let data = vec![u32::MAX, 1, 2, 3, 4];
    let gpu_array = UInt32ArrayGPU::from_vec(&data, device.clone());
    println!("{:?}", gpu_array.add(&100).await);

    let data = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let mut gpu_array = Float32ArrayGPU::from_vec(&data, device.clone());
    println!("{:?}", gpu_array.add(&200.0).await)
}
