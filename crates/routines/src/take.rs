use arrow_gpu_array::array::{ArrowArrayGPU, UInt32ArrayGPU};
use arrow_gpu_array::gpu_utils::*;
use wgpu::Buffer;

use crate::Swizzle;

pub(crate) const U32_TAKE_SHADER: &str = include_str!("../compute_shaders/32bit/take.wgsl");

pub(crate) fn apply_take_op(
    device: &GpuDevice,
    operand_1: &Buffer,
    operand_2: &Buffer,
    dispatch_size: u64,
    output_size: u64,
    shader: &str,
    entry_point: &str,
    pipeline: &mut ArrowComputePipeline,
) -> Buffer {
    let compute_pipeline = device.create_compute_pipeline(shader, entry_point);

    let new_values_buffer = device.create_empty_buffer(output_size);

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

    let query = device.compute_pass(
        &mut pipeline.encoder,
        None,
        &compute_pipeline,
        &bind_group_array,
        entry_point,
        dispatch_size.div_ceil(256) as u32,
    );

    query.resolve(&mut pipeline.encoder);
    pipeline.queries.push(query);
    new_values_buffer
}

pub fn take_dyn(operand_1: &ArrowArrayGPU, indexes: &UInt32ArrayGPU) -> ArrowArrayGPU {
    let mut pipeline = ArrowComputePipeline::new(operand_1.get_gpu_device(), Some("take"));
    let result = take_op_dyn(operand_1, indexes, &mut pipeline);
    pipeline.finish();
    result
}

macro_rules! take_op_dyn_arms {
    ($operand_1: ident, $indexes: ident, $pipeline:ident, $($arr: ident),*) => {
        match $operand_1 {
            $(ArrowArrayGPU::$arr(op1) => {
                op1.take_op($indexes, $pipeline).into()
            })*
            _ => panic!(
                "Take Operation not supported for {:?}",
                $operand_1.get_dtype(),
            ),
        }
    };
}

pub fn take_op_dyn(
    operand_1: &ArrowArrayGPU,
    indexes: &UInt32ArrayGPU,
    pipeline: &mut ArrowComputePipeline,
) -> ArrowArrayGPU {
    take_op_dyn_arms!(
        operand_1,
        indexes,
        pipeline,
        Date32ArrayGPU,
        UInt32ArrayGPU,
        Int32ArrayGPU,
        Float32ArrayGPU,
        BooleanArrayGPU
    )
}

#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_take_op {
        ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::gpu_utils::GpuDevice;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand1_type::from_slice(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_slice(&$input_2, device.clone());
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2);
                assert_eq!(new_gpu_array.raw_values().unwrap(), $output);
            }
        };
        ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::gpu_utils::GpuDevice;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_slice(&$input_2, device.clone());
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
