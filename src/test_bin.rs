use arrow_gpu::array::gpu_array::primitive_array_gpu::PrimitiveArrayGpu;
use arrow_gpu::kernels::add_scalar::AddScalarKernel;

#[tokio::main]
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

    let data = vec![u32::MAX, 1, 2, 3, 4];
    let gpu_array = PrimitiveArrayGpu::<u32>::new(data);

    println!("{:?}", gpu_array.add_scalar(100));

    let data = vec![0i32, 1, 2, 3, 4];
    let gpu_array = PrimitiveArrayGpu::<i32>::new(data);
    gpu_array.add_scalar(-10);

    let data = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let gpu_array = PrimitiveArrayGpu::<f32>::new(data);
    gpu_array.add_scalar(10.0);

    let data = vec![0u16, 1, 2, 3, 4];
    let gpu_array = PrimitiveArrayGpu::<u16>::new(data);

    println!("{:?}", gpu_array);
    println!("{:?}", gpu_array.add_scalar(100));

    //gpu_array.add_scalar(10);
}
