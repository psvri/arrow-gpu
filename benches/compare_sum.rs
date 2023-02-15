use std::sync::Arc;

use arrow::array::Float32Array;
use arrow::compute::kernels::aggregate::sum;
use arrow_gpu::{
    array::{f32_gpu::Float32ArrayGPU, gpu_device::GpuDevice},
    kernels::aggregate::ArrowSum,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pollster::FutureExt;

fn bench_cpu_f32_add(data: &mut Float32Array) -> f32 {
    sum(data).unwrap()
}

fn bench_gpu_f32_add(data: &mut Float32ArrayGPU) -> f32 {
    data.sum().block_on()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let device = GpuDevice::new().block_on();
    let mut gpu_data = Float32ArrayGPU::from_vec(
        &(0..1024 * 1024 * 10)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        Arc::new(device),
    );
    let mut cpu_data = Float32Array::from(
        (0..1024 * 1024 * 10)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
    );
    c.bench_function("sum gpu f32", |b| {
        b.iter(|| bench_gpu_f32_add(black_box(&mut gpu_data)))
    });
    c.bench_function("sum cpu f32", |b| {
        b.iter(|| bench_cpu_f32_add(black_box(&mut cpu_data)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
