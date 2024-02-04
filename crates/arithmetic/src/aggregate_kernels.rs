use std::sync::Arc;

use arrow_gpu_array::{
    array::{ArrayUtils, UInt32ArrayGPU},
    gpu_utils::ArrowComputePipeline,
};

pub trait Sum: ArrayUtils + Sized {
    fn sum(&self) -> Self {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), Some("Sum"));
        let result = self.sum_op(&mut pipeline);
        pipeline.finish();
        result
    }

    fn sum_op(&self, pipeline: &mut ArrowComputePipeline) -> Self;
}

const SHADER: &str = include_str!("../compute_shaders/u32/aggregate.wgsl");

impl Sum for UInt32ArrayGPU {
    fn sum_op(&self, pipeline: &mut ArrowComputePipeline) -> Self {
        let mut new_length = self.len.div_ceil(256);
        let mut temp_buffer = pipeline.apply_unary_function(
            &self.data,
            (new_length * 4) as u64,
            SHADER,
            "sum",
            new_length as u32,
        );
        while new_length != 1 {
            new_length = new_length.div_ceil(256);
            temp_buffer = pipeline.apply_unary_function(
                &temp_buffer,
                (new_length * 4) as u64,
                SHADER,
                "sum",
                new_length as u32,
            );
        }
        UInt32ArrayGPU {
            data: Arc::new(temp_buffer),
            gpu_device: self.get_gpu_device(),
            phantom: std::marker::PhantomData,
            len: 1,
            null_buffer: None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu_array::{gpu_utils::GpuDevice, GPU_DEVICE};

    #[test]
    fn test_sum() {
        let device = GPU_DEVICE
            .get_or_init(|| Arc::new(GpuDevice::new()))
            .clone();
        let array = UInt32ArrayGPU::broadcast(2, 256 * 256, device);
        assert_eq!(array.sum().raw_values().unwrap(), vec![2 * 256 * 256]);
    }

    #[test]
    fn test_sum_large() {
        let device = GPU_DEVICE
            .get_or_init(|| Arc::new(GpuDevice::new()))
            .clone();
        let array = UInt32ArrayGPU::broadcast(2, 1 * 1024 * 1024, device);
        assert_eq!(array.sum().raw_values().unwrap(), vec![2 * 1024 * 1024]);
    }
}
