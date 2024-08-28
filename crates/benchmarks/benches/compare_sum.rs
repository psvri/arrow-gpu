use std::sync::Arc;

use arrow::array::UInt32Array;
use arrow::compute::kernels::aggregate::sum;
use arrow_gpu::kernels::broadcast::Broadcast;
use arrow_gpu::kernels::Sum;
use arrow_gpu::{array::*, gpu_utils::*};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_cpu_u32_add(data: &mut UInt32Array) -> u32 {
    sum(data).unwrap()
}

fn bench_gpu_u32_add(data: &mut UInt32ArrayGPU) -> UInt32ArrayGPU {
    data.sum()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let device = Arc::new(GpuDevice::new());

    let size = 1 * 1024 * 1024;
    let base_value = 2;

    let mut group = c.benchmark_group("u32_array_sum");

    for i in [1, 10] {
        let input_size = size * i;

        let mut gpu_data = UInt32ArrayGPU::broadcast(base_value, input_size, device.clone());
        let mut cpu_data = UInt32Array::from(vec![base_value; input_size]);

        group.throughput(Throughput::Bytes(input_size as u64));

        group.bench_function(format!("sum gpu u32 {} million", i), |b| {
            b.iter(|| bench_gpu_u32_add(black_box(&mut gpu_data)))
        });
        group.bench_function(format!("sum cpu u32 {} million", i), |b| {
            b.iter(|| bench_cpu_u32_add(black_box(&mut cpu_data)))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
