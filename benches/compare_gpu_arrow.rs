use std::sync::Arc;

use arrow::array::Float32Array;
use arrow::compute::kernels::arithmetic::add_scalar;
use arrow_gpu::{
    array::gpu_array::{f32_gpu::Float32ArrayGPU, GpuDevice},
    kernels::arithmetic::*,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pollster::FutureExt;

fn bench_cpu_f32_add(data: &mut Float32Array, value: f32) -> Float32Array {
    add_scalar(data, value).unwrap()
}

fn bench_gpu_f32_add(data: &mut Float32ArrayGPU, value: f32) -> Float32ArrayGPU {
    data.add(&value).block_on()
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
    c.bench_function("gpu f32", |b| {
        b.iter(|| bench_gpu_f32_add(black_box(&mut gpu_data), black_box(100.0)))
    });
    c.bench_function("cpu f32", |b| {
        b.iter(|| bench_cpu_f32_add(black_box(&mut cpu_data), black_box(100.0)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
