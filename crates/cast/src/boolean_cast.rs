use std::sync::Arc;

use crate::Cast;
use arrow_gpu_array::array::{BooleanArrayGPU, Float32ArrayGPU, NullBitBufferGpu};
use arrow_gpu_array::gpu_utils::*;
use wgpu::Buffer;

const BOOLEAN_CAST_F32_SHADER: &str = include_str!("../compute_shaders/boolean/cast_f32.wgsl");

pub fn apply_boolean_unary_function(
    gpu_device: &GpuDevice,
    original_values: &Buffer,
    new_buffer_size: u64,
    output_item_size: u64,
    shader: &str,
    entry_point: &str,
    pipeline: &mut ArrowComputePipeline,
) -> Buffer {
    let compute_pipeline = gpu_device.create_compute_pipeline(shader, entry_point);

    let new_values_buffer = gpu_device.create_empty_buffer(new_buffer_size);

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group_array = gpu_device
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: original_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

    let dispatch_size = new_buffer_size.div_ceil(output_item_size);

    let query = gpu_device.compute_pass(
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

impl Cast<Float32ArrayGPU> for BooleanArrayGPU {
    fn cast_op(&self, pipeline: &mut ArrowComputePipeline) -> Float32ArrayGPU {
        let new_buffer = apply_boolean_unary_function(
            &self.gpu_device,
            &self.data,
            self.len as u64 * 4,
            4,
            BOOLEAN_CAST_F32_SHADER,
            "cast_f32",
            pipeline,
        );

        Float32ArrayGPU {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            phantom: Default::default(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::cast_dyn;
    use crate::tests::test_cast_op;
    use crate::ArrowType;

    test_cast_op!(
        test_cast_boolean_to_f32,
        BooleanArrayGPU,
        Float32ArrayGPU,
        [true, false, true, true, false, false, true, true, false],
        Float32Type,
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
}
