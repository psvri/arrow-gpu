use arrow_gpu_array::array::{ArrowArrayGPU, BooleanArrayGPU};
use arrow_gpu_array::gpu_utils::*;
use wgpu::Buffer;

pub(crate) const U32_MERGE_SHADER: &str = include_str!("../compute_shaders/32bit/merge.wgsl");
pub(crate) const U16_MERGE_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/16bit/merge.wgsl")
);
pub(crate) const U8_MERGE_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/8bit/merge.wgsl")
);

use crate::Swizzle;

pub fn merge_null_buffers(
    device: &GpuDevice,
    operand_1_null_buffer: Option<&Buffer>,
    operand_2_null_buffer: Option<&Buffer>,
    mask: &Buffer,
    mask_null_buffer: Option<&Buffer>,
) -> Option<Buffer> {
    const SHADER: &str = include_str!("../compute_shaders/u32/merge_null_buffer.wgsl");

    let merged_buffer_1 = operand_1_null_buffer.map(|op1_null_buffer| {
        device.apply_binary_function(op1_null_buffer, mask, 4, SHADER, "merge_selected")
    });

    let merged_buffer_2 = operand_2_null_buffer.map(|op2_null_buffer| {
        device.apply_binary_function(op2_null_buffer, mask, 4, SHADER, "merge_not_selected")
    });

    let merged_buffer = match (merged_buffer_1, merged_buffer_2) {
        (Some(mb1), Some(mb2)) => {
            Some(device.apply_binary_function(&mb1, &mb2, 4, SHADER, "merge_or"))
        }
        (None, Some(mb)) | (Some(mb), None) => Some(mb),
        (None, None) => None,
    };

    match (merged_buffer, mask_null_buffer) {
        (Some(mb1), Some(mb2)) => {
            Some(device.apply_binary_function(&mb1, mb2, 4, SHADER, "merge_nulls"))
        }
        (None, Some(mb)) => Some(device.clone_buffer(mb)),
        (Some(mb), None) => Some(mb),
        (None, None) => None,
    }
}

pub fn merge_null_buffers_op(
    operand_1_null_buffer: Option<&Buffer>,
    operand_2_null_buffer: Option<&Buffer>,
    mask: &Buffer,
    mask_null_buffer: Option<&Buffer>,
    pipeline: &mut ArrowComputePipeline,
) -> Option<Buffer> {
    const SHADER: &str = include_str!("../compute_shaders/u32/merge_null_buffer.wgsl");

    let merged_buffer_1 = if let Some(op1_null_buffer) = operand_1_null_buffer {
        let dispatch_size = op1_null_buffer.size().div_ceil(4).div_ceil(256) as u32;
        Some(pipeline.apply_binary_function(
            op1_null_buffer,
            mask,
            op1_null_buffer.size(),
            SHADER,
            "merge_selected",
            dispatch_size,
        ))
    } else {
        None
    };

    let merged_buffer_2 = if let Some(op2_null_buffer) = operand_2_null_buffer {
        let dispatch_size = op2_null_buffer.size().div_ceil(4).div_ceil(256) as u32;
        Some(pipeline.apply_binary_function(
            op2_null_buffer,
            mask,
            op2_null_buffer.size(),
            SHADER,
            "merge_not_selected",
            dispatch_size,
        ))
    } else {
        None
    };

    let merged_buffer = match (merged_buffer_1, merged_buffer_2) {
        (Some(mb1), Some(mb2)) => {
            let dispatch_size = mb1.size().div_ceil(4).div_ceil(256) as u32;
            Some(pipeline.apply_binary_function(
                &mb1,
                &mb2,
                mb1.size(),
                SHADER,
                "merge_or",
                dispatch_size,
            ))
        }
        (None, Some(mb)) | (Some(mb), None) => Some(mb),
        (None, None) => None,
    };

    match (merged_buffer, mask_null_buffer) {
        (Some(mb1), Some(mb2)) => {
            let dispatch_size = mb1.size().div_ceil(4).div_ceil(256) as u32;
            Some(pipeline.apply_binary_function(
                &mb1,
                mb2,
                mb1.size(),
                SHADER,
                "merge_nulls",
                dispatch_size,
            ))
        }
        (None, Some(mb)) => Some(pipeline.clone_buffer(mb)),
        (Some(mb), None) => Some(mb),
        (None, None) => None,
    }
}

pub fn merge_dyn(
    operand_1: &ArrowArrayGPU,
    operand_2: &ArrowArrayGPU,
    mask: &BooleanArrayGPU,
) -> ArrowArrayGPU {
    match (operand_1, operand_2) {
        (ArrowArrayGPU::Date32ArrayGPU(op1), ArrowArrayGPU::Date32ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(op1), ArrowArrayGPU::Int32ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(op1), ArrowArrayGPU::Int16ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(op1), ArrowArrayGPU::Int8ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(op1), ArrowArrayGPU::UInt32ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(op1), ArrowArrayGPU::UInt16ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(op1), ArrowArrayGPU::UInt8ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        (ArrowArrayGPU::Float32ArrayGPU(op1), ArrowArrayGPU::Float32ArrayGPU(op2)) => {
            op1.merge(op2, mask).into()
        }
        _ => panic!(
            "Merge Operation not supported between {:?} and {:?}",
            operand_1.get_dtype(),
            operand_2.get_dtype()
        ),
    }
}

pub fn merge_op_dyn(
    operand_1: &ArrowArrayGPU,
    operand_2: &ArrowArrayGPU,
    mask: &BooleanArrayGPU,
    pipeline: &mut ArrowComputePipeline,
) -> ArrowArrayGPU {
    match (operand_1, operand_2) {
        (ArrowArrayGPU::Date32ArrayGPU(op1), ArrowArrayGPU::Date32ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::Int32ArrayGPU(op1), ArrowArrayGPU::Int32ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(op1), ArrowArrayGPU::Int16ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(op1), ArrowArrayGPU::Int8ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::UInt32ArrayGPU(op1), ArrowArrayGPU::UInt32ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(op1), ArrowArrayGPU::UInt16ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(op1), ArrowArrayGPU::UInt8ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        (ArrowArrayGPU::Float32ArrayGPU(op1), ArrowArrayGPU::Float32ArrayGPU(op2)) => {
            op1.merge_op(op2, mask, pipeline).into()
        }
        _ => panic!(
            "Merge Operation not supported between {:?} and {:?}",
            operand_1.get_dtype(),
            operand_2.get_dtype()
        ),
    }
}

#[cfg(test)]
mod test {
    #[macro_export]
    macro_rules! test_merge_op {
        ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $mask: expr, $output: expr) => {
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::gpu_utils::GpuDevice;
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device.clone());
                let mask = BooleanArrayGPU::from_optional_slice(&$mask, device.clone());
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2, &mask);
                assert_eq!(new_gpu_array.values(), $output);
            }
        };
        ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr,  $mask: expr, $output: expr) => {
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::gpu_utils::GpuDevice;
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.get_or_init(|| Arc::new(GpuDevice::new()).clone());
                let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
                let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device.clone());
                let mask = BooleanArrayGPU::from_optional_slice(&$mask, device.clone());
                let new_gpu_array = gpu_array_1.$operation(&gpu_array_2, &mask);
                assert_eq!(new_gpu_array.values(), $output);

                let new_gpu_array = $operation_dyn(&gpu_array_1.into(), &gpu_array_2.into(), &mask);
                let new_values = $output_type::try_from(new_gpu_array).unwrap().values();
                assert_eq!(new_values, $output);
            }
        };
    }
}
