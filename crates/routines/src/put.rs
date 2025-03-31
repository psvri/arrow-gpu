use arrow_gpu_array::array::{ArrowArrayGPU, UInt32ArrayGPU};
use arrow_gpu_array::gpu_utils::*;
use wgpu::Buffer;

use crate::Swizzle;

pub(crate) const U32_PUT_SHADER: &str = include_str!("../compute_shaders/32bit/put.wgsl");

pub(crate) fn apply_put_op(
    device: &GpuDevice,
    src_buffer: &Buffer,
    dst_buffer: &Buffer,
    src_indexes: &Buffer,
    dst_indexes: &Buffer,
    dispatch_size: u64,
    shader: &str,
    entry_point: &str,
    pipeline: &mut ArrowComputePipeline,
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

    let query = device.compute_pass(
        &mut pipeline.encoder,
        None,
        &compute_pipeline,
        &bind_group_array,
        entry_point,
        dispatch_size.div_ceil(256) as u32,
    );

    query.resolve(&mut pipeline.encoder);
}

/// Put elements from src using src_indexes into dst using dst_indexes
pub fn put_dyn(
    src: &ArrowArrayGPU,
    src_indexes: &UInt32ArrayGPU,
    dst: &mut ArrowArrayGPU,
    dst_indexes: &UInt32ArrayGPU,
) {
    let mut pipeline = ArrowComputePipeline::new(src.get_gpu_device(), Some("put"));
    put_op_dyn(src, src_indexes, dst, dst_indexes, &mut pipeline);
    pipeline.finish();
}

macro_rules! put_op_dyn_arms {
    ($operand_1: ident, $operand_2: ident, $src_indexes: ident, $dst_indexes: ident, $pipeline:ident, $($arr: ident),*) => {
        match ($operand_1, $operand_2) {
            $((ArrowArrayGPU::$arr(op1), ArrowArrayGPU::$arr(op2)) => {
                op1.put_op($src_indexes, op2, $dst_indexes, $pipeline).into()
            })*
            // temporary workaround as 2024 edition gives error
            // the trait bound `!: From<()>` is not satisfied
            #[allow(unreachable_code)]
            (x, y) => panic!(
                "Put Operation not supported for {:?} and {:?}",
                x.get_dtype(),
                y.get_dtype(),
            ) as (),
        }
    };
}

/// Submits a command to put elements from src using src_indexes into dst using dst_indexes
pub fn put_op_dyn(
    src: &ArrowArrayGPU,
    src_indexes: &UInt32ArrayGPU,
    dst: &mut ArrowArrayGPU,
    dst_indexes: &UInt32ArrayGPU,
    pipeline: &mut ArrowComputePipeline,
) {
    put_op_dyn_arms!(
        src,
        dst,
        src_indexes,
        dst_indexes,
        pipeline,
        Float32ArrayGPU,
        Int32ArrayGPU,
        UInt32ArrayGPU,
        Date32ArrayGPU,
        BooleanArrayGPU
    );
}

#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_put_op {
        ($(#[$m:meta])* $fn_name: ident, $operand_type: ident, $operation: ident, $src: expr, $dst: expr, $src_index: expr, $dst_index: expr, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                use arrow_gpu_array::gpu_utils::GpuDevice;
                use pollster::FutureExt;
                let device = GPU_DEVICE.clone();
                let gpu_array_1 = $operand_type::from_slice(&$src, device.clone());
                let mut gpu_array_2 = $operand_type::from_slice(&$dst, device.clone());
                let src_index = UInt32ArrayGPU::from_slice(&$src_index, device.clone());
                let dst_index = UInt32ArrayGPU::from_slice(&$dst_index, device);
                gpu_array_1.$operation(&src_index, &mut gpu_array_2, &dst_index);
                assert_eq!(gpu_array_2.raw_values().unwrap(), $output);
            }
        };
        ($(#[$m:meta])* $fn_name: ident, $operand_type: ident, $operation: ident, $operation_dyn: ident, $src: expr, $dst: expr, $src_index: expr, $dst_index: expr, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.clone();
                let gpu_array_1 = $operand_type::from_slice(&$src, device.clone());
                let mut gpu_array_2 = $operand_type::from_slice(&$dst, device.clone());
                let src_index = UInt32ArrayGPU::from_slice(&$src_index, device.clone());
                let dst_index = UInt32ArrayGPU::from_slice(&$dst_index, device.clone());
                gpu_array_1.$operation(&src_index, &mut gpu_array_2, &dst_index);
                assert_eq!(gpu_array_2.raw_values().unwrap(), $output);

                let mut gpu_array_2_dyn = $operand_type::from_slice(&$dst, device.clone()).into();
                $operation_dyn(&gpu_array_1.into(), &src_index, &mut gpu_array_2_dyn, &dst_index);

                let new_values = $operand_type::try_from(gpu_array_2_dyn)
                    .unwrap()
                    .raw_values()
                    .unwrap();
                assert_eq!(new_values, $output);
            }
        };
    }
}
