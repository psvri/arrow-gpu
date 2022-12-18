use arrow_gpu::array::gpu_array::primitive_array_gpu::PrimitiveArrayGpu;
use arrow_gpu::kernels::add_ops::{ArrowAdd, ArrowAddAssign};

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

    let data = vec![u32::MAX, 1, 2, 3, 4];
    let mut gpu_array = PrimitiveArrayGpu::<u32>::new(data);
    gpu_array.add_assign(100).await;

    //println!("{:?}", gpu_array.add_assign(100).await);
    println!("{:?}", gpu_array);

    let data = vec![0i32, 1, 2, 3, 4];
    let mut gpu_array = PrimitiveArrayGpu::<i32>::new(data);
    gpu_array.add_assign(-10).await;

    let data = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let mut gpu_array = PrimitiveArrayGpu::<f32>::new(data);
    gpu_array.add_assign(10.0).await;
    println!("{:?}", gpu_array.add(200.0).await);

    let data = vec![0u16, 1, 2, 3, 4];
    let mut gpu_array = PrimitiveArrayGpu::<u16>::new(data);

    println!("{:?}", gpu_array);
    println!("{:?}", gpu_array.add_assign(100).await);
}
