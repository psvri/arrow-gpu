use arrow_gpu_array::array::{ArrowArrayGPU, GpuDevice, UInt32ArrayGPU};
use wgpu::Buffer;

use crate::Swizzle;

pub(crate) const U32_TAKE_SHADER: &str = include_str!("../compute_shaders/32bit/take.wgsl");

pub(crate) fn apply_take_function(
    device: &GpuDevice,
    operand_1: &Buffer,
    operand_2: &Buffer,
    output_count: u64,
    item_size: u64,
    shader: &str,
    entry_point: &str,
) -> Buffer {
    let compute_pipeline = device.create_compute_pipeline(shader, entry_point);

    let new_values_buffer = device.create_empty_buffer(output_count * item_size);

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group_array = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: operand_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: operand_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: new_values_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(Some(entry_point));
    let dispatch_size = output_count;

    let query = device.compute_pass(
        &mut encoder,
        None,
        &compute_pipeline,
        &bind_group_array,
        entry_point,
        dispatch_size.div_ceil(256) as u32,
    );

    query.resolve(&mut encoder);
    device.queue.submit(Some(encoder.finish()));
    new_values_buffer
}

pub fn take_dyn(operand_1: &ArrowArrayGPU, indexes: &UInt32ArrayGPU) -> ArrowArrayGPU {
    match operand_1 {
        ArrowArrayGPU::Date32ArrayGPU(op1) => op1.take(indexes).into(),
        ArrowArrayGPU::UInt32ArrayGPU(op1) => op1.take(indexes).into(),
        ArrowArrayGPU::Int32ArrayGPU(op1) => op1.take(indexes).into(),
        ArrowArrayGPU::Float32ArrayGPU(op1) => op1.take(indexes).into(),
        _ => panic!(
            "Take Operation not supported for {:?}",
            operand_1.get_dtype(),
        ),
    }
}

#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_take_op {
        ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            $(#[$m])*
            #[tokio::test]
            async fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::array::GpuDevice;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand1_type::from_slice(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_slice(&$input_2, device.clone());
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2);
                assert_eq!(new_gpu_array.raw_values().unwrap(), $output);
            }
        };
        ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            $(#[$m])*
            #[tokio::test]
            async fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::array::GpuDevice;
                use pollster::FutureExt;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new().block_on()).clone());
                let gpu_array_1 = $operand1_type::from_optional_vec(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_optional_vec(&$input_2, device);
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2);
                assert_eq!(new_gpu_array.values(), $output);

                let new_gpu_array = $operation_dyn(&gpu_array_1.into(), &gpu_array_2.into());
                let new_values = $output_type::try_from(new_gpu_array)
                    .unwrap()
                    .values();
                assert_eq!(new_values, $output);
            }
        };
    }
}
