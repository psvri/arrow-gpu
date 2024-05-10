use std::sync::Arc;

use arrow_gpu_array::{
    array::{BooleanArrayGPU, NullBitBufferGpu, UInt32ArrayGPU},
    gpu_utils::{ArrowComputePipeline, GpuDevice},
};
use wgpu::Buffer;

use crate::{merge_null_buffers_op, put::apply_put_op, take::apply_take_op, Swizzle};

const MERGE_SHADER: &str = include_str!("../compute_shaders/bool/merge.wgsl");
const PUT_SHADER: &str = include_str!("../compute_shaders/bool/put.wgsl");
const TAKE_SHADER: &str = include_str!("../compute_shaders/bool/take.wgsl");

pub(crate) fn take_bool(
    device: &GpuDevice,
    data: &Buffer,
    indexes: &UInt32ArrayGPU,
    pipeline: &mut ArrowComputePipeline,
) -> Buffer {
    apply_take_op(
        device,
        data,
        &indexes.data,
        indexes.len as u64,
        (indexes.len.div_ceil(32) as u64) * 4,
        TAKE_SHADER,
        "take",
        pipeline,
    )
}

pub(crate) fn take_null_buffer(
    data: Option<&NullBitBufferGpu>,
    indexes: &UInt32ArrayGPU,
    pipeline: &mut ArrowComputePipeline,
) -> Option<NullBitBufferGpu> {
    data.map(|x| {
        let new_bit_bufer = take_bool(&x.gpu_device, &x.bit_buffer, indexes, pipeline);
        NullBitBufferGpu {
            bit_buffer: Arc::new(new_bit_bufer),
            len: x.len,
            gpu_device: x.gpu_device.clone(),
        }
    })
}

impl Swizzle for BooleanArrayGPU {
    fn merge_op(
        &self,
        other: &Self,
        mask: &BooleanArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) -> Self {
        let new_buffer = pipeline.apply_ternary_function(
            &self.data,
            &other.data,
            &mask.data,
            4,
            MERGE_SHADER,
            "merge_array",
        );

        let op1 = self.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let op2 = other.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());
        let mask_null = mask.null_buffer.as_ref().map(|x| x.bit_buffer.as_ref());

        //TODO can be simplified
        let bit_buffer = merge_null_buffers_op(op1, op2, &mask.data, mask_null, pipeline);

        let new_null_buffer = bit_buffer.map(|buffer| NullBitBufferGpu {
            bit_buffer: Arc::new(buffer),
            len: self.len,
            gpu_device: self.gpu_device.clone(),
        });

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: new_null_buffer,
        }
    }

    fn take_op(&self, indexes: &UInt32ArrayGPU, pipeline: &mut ArrowComputePipeline) -> Self {
        let new_buffer = take_bool(&self.gpu_device, &self.data, indexes, pipeline);

        let null_buffer = take_null_buffer(self.null_buffer.as_ref(), indexes, pipeline);

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: indexes.len,
            null_buffer,
        }
    }

    fn put_op(
        &self,
        src_indexes: &UInt32ArrayGPU,
        dst: &mut Self,
        dst_indexes: &UInt32ArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) {
        apply_put_op(
            &self.gpu_device,
            &self.data,
            &dst.data,
            &src_indexes.data,
            &dst_indexes.data,
            (src_indexes.len as u64).div_ceil(32),
            PUT_SHADER,
            "put",
            pipeline,
        );

        match (&self.null_buffer, &dst.null_buffer) {
            (None, None) => {}
            (None, Some(_)) => todo!(),
            (Some(_), None) => todo!(),
            (Some(_), Some(_)) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use arrow_gpu_array::array::BooleanArrayGPU;

    test_merge_op!(
        test_merge_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        merge,
        merge_dyn,
        [
            Some(true),
            Some(true),
            None,
            None,
            Some(true),
            Some(true),
            Some(true),
            None,
            Some(true)
        ],
        [
            Some(false),
            Some(false),
            None,
            Some(false),
            None,
            None,
            Some(false),
            Some(false),
            None
        ],
        [
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            None,
            None,
            Some(false),
        ],
        [
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            None,
            None,
            None
        ]
    );

    // TODO test for cases with null
    test_put_op!(
        test_put_bool,
        BooleanArrayGPU,
        put,
        put_dyn,
        [true, true, false, false],
        [true; 8],
        [0, 1, 2, 3],
        [1, 3, 5, 7],
        [true, true, true, true, true, false, true, false]
    );

    // TODO test for cases with null
    test_take_op!(
        test_take_bool,
        BooleanArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        take,
        [true, true, false, false],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [true, true, false, false, true, true, false, false]
    );

    // TODO test for cases with null
    test_take_op!(
        test_large_take_bool,
        BooleanArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        take,
        [true],
        [0; 100],
        [true; 100]
    );
}
