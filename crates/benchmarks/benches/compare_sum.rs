use std::sync::Arc;

use arrow::array::UInt32Array;
use arrow::compute::kernels::aggregate::sum;
use arrow_gpu::kernels::Sum;
use arrow_gpu::{array::*, gpu_utils::*};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_cpu_u32_add(data: &mut UInt32Array) -> u32 {
    sum(data).unwrap()
}

fn bench_gpu_u32_add(data: &mut UInt32ArrayGPU) -> UInt32ArrayGPU {
    data.sum()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let device = GpuDevice::new();

    let input = vec![1; 1 * 1024 * 1024];

    let mut gpu_data = UInt32ArrayGPU::from_slice(&input, Arc::new(device));
    let mut cpu_data = UInt32Array::from(input);
    c.bench_function("sum gpu u32", |b| {
        b.iter(|| bench_gpu_u32_add(black_box(&mut gpu_data)))
    });
    c.bench_function("sum cpu u32", |b| {
        b.iter(|| bench_cpu_u32_add(black_box(&mut cpu_data)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
