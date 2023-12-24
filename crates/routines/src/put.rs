use arrow_gpu_array::array::{ArrowArrayGPU, GpuDevice, UInt32ArrayGPU};
use wgpu::{Buffer, Maintain};

use crate::Swizzle;

pub(crate) const U32_PUT_SHADER: &str = include_str!("../compute_shaders/32bit/put.wgsl");

pub(crate) async fn apply_put_function(
    device: &GpuDevice,
    src_buffer: &Buffer,
    dst_buffer: &Buffer,
    src_indexes: &Buffer,
    dst_indexes: &Buffer,
    indexes_count: u64,
    shader: &str,
    entry_point: &str,
) {
    let compute_pipeline = device.create_compute_pipeline(shader, entry_point);

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group_array = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: src_indexes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dst_indexes.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(Some(entry_point));
    let dispatch_size = indexes_count;

    let query = device.compute_pass(
        &mut encoder,
        None,
        &compute_pipeline,
        &bind_group_array,
        entry_point,
        dispatch_size.div_ceil(256) as u32,
    );

    query.resolve(&mut encoder);
    let submission_index = device.queue.submit(Some(encoder.finish()));
    device
        .device
        .poll(Maintain::WaitForSubmissionIndex(submission_index));
    query.wait_for_results(&device.device, &device.queue).await;
}

pub async fn put_dyn(
    src: &ArrowArrayGPU,
    src_indexes: &UInt32ArrayGPU,
    dst: &mut ArrowArrayGPU,
    dst_indexes: &UInt32ArrayGPU,
) {
    match (src, dst) {
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowArrayGPU::Float32ArrayGPU(y)) => {
            x.put(src_indexes, y, dst_indexes).await
        }
        (ArrowArrayGPU::Int32ArrayGPU(x), ArrowArrayGPU::Int32ArrayGPU(y)) => {
            x.put(src_indexes, y, dst_indexes).await
        }
        (ArrowArrayGPU::UInt32ArrayGPU(x), ArrowArrayGPU::UInt32ArrayGPU(y)) => {
            x.put(src_indexes, y, dst_indexes).await
        }
        (ArrowArrayGPU::Date32ArrayGPU(x), ArrowArrayGPU::Date32ArrayGPU(y)) => {
            x.put(src_indexes, y, dst_indexes).await
        }
        (x, y) => panic!(
            "Put Operation not supported for {:?} and {:?}",
            x.get_dtype(),
            y.get_dtype(),
        ),
    }
}

#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_put_op {
        ($(#[$m:meta])* $fn_name: ident, $operand_type: ident, $operation: ident, $src: expr, $dst: expr, $src_index: expr, $dst_index: expr, $output: expr) => {
            $(#[$m])*
            #[tokio::test]
            async fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::array::GpuDevice;
                use pollster::FutureExt;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand_type::from_slice(&$src, device.clone());
                let mut gpu_array_2 = $operand_type::from_slice(&$dst, device.clone());
                let src_index = UInt32ArrayGPU::from_slice(&$src_index, device.clone());
                let dst_index = UInt32ArrayGPU::from_slice(&$dst_index, device);
                gpu_array_1.$operation(&src_index, &mut gpu_array_2, &dst_index).await;
                assert_eq!(gpu_array_2.raw_values().unwrap(), $output);
            }
        };
        ($(#[$m:meta])* $fn_name: ident, $operand_type: ident, $operation: ident, $operation_dyn: ident, $src: expr, $dst: expr, $src_index: expr, $dst_index: expr, $output: expr) => {
            $(#[$m])*
            #[tokio::test]
            async fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::array::GpuDevice;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand_type::from_slice(&$src, device.clone());
                let mut gpu_array_2 = $operand_type::from_slice(&$dst, device.clone());
                let src_index = UInt32ArrayGPU::from_slice(&$src_index, device.clone());
                let dst_index = UInt32ArrayGPU::from_slice(&$dst_index, device.clone());
                gpu_array_1.$operation(&src_index, &mut gpu_array_2, &dst_index).await;
                assert_eq!(gpu_array_2.raw_values().unwrap(), $output);

                let mut gpu_array_2_dyn = $operand_type::from_slice(&$dst, device.clone()).into();
                $operation_dyn(&gpu_array_1.into(), &src_index, &mut gpu_array_2_dyn, &dst_index).await;

                let new_values = $operand_type::try_from(gpu_array_2_dyn)
                    .unwrap()
                    .raw_values()
                    .unwrap();
                assert_eq!(new_values, $output);
            }
        };
    }
}
