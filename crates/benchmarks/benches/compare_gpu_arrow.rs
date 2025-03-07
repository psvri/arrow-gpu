use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Datum, Float32Array},
    compute::kernels::numeric::add,
};
use arrow_gpu::{array::*, gpu_utils::*, kernels::add_dyn};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_cpu_f32_add(data: &mut Float32Array, value: &dyn Datum) -> ArrayRef {
    add(data, value).unwrap()
}

fn bench_gpu_f32_add(data: &ArrowArrayGPU, value: &ArrowArrayGPU) -> ArrowArrayGPU {
    add_dyn(data, value)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new());
    let count = 1024 * 1024 * 10;
    let mut gpu_data = Float32ArrayGPU::from_slice(
        &(0..count)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
        device.clone(),
    )
    .into();
    let value_data = Float32ArrayGPU::from_slice(&[100.0], device.clone()).into();
    let mut cpu_data = Float32Array::from(
        (0..count)
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>(),
    );
    let cpu_value = Float32Array::new_scalar(100.0);
    c.bench_function("gpu f32", |b| {
        b.iter(|| bench_gpu_f32_add(black_box(&mut gpu_data), black_box(&value_data)))
    });
    c.bench_function("cpu f32", |b| {
        b.iter(|| bench_cpu_f32_add(black_box(&mut cpu_data), black_box(&cpu_value)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
