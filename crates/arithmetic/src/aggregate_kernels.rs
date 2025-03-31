use arrow_gpu_array::{
    array::{ArrayUtils, ArrowPrimitiveType, PrimitiveArrayGpu},
    gpu_utils::ArrowComputePipeline,
};
use std::sync::Arc;

/// Trait for sum of all elements in the array
pub trait Sum: ArrayUtils + Sized {
    fn sum(&self) -> Self {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), Some("Sum"));
        let result = self.sum_op(&mut pipeline);
        pipeline.finish();
        result
    }

    /// Computes sum of all elements in the array
    fn sum_op(&self, pipeline: &mut ArrowComputePipeline) -> Self;
}

/// Helper trait for Arrow arrays backed by 32 bits that support sum
pub trait Sum32Bit: ArrowPrimitiveType {
    const SHADER: &'static str;
}

impl<T: Sum32Bit> Sum for PrimitiveArrayGpu<T> {
    fn sum_op(&self, pipeline: &mut ArrowComputePipeline) -> Self {
        let mut new_length = self.len.div_ceil(256);
        let mut temp_buffer = pipeline.apply_unary_function(
            &self.data,
            (new_length * 4) as u64,
            T::SHADER,
            "sum",
            new_length as u32,
        );
        while new_length != 1 {
            new_length = new_length.div_ceil(256);
            temp_buffer = pipeline.apply_unary_function(
                &temp_buffer,
                (new_length * 4) as u64,
                T::SHADER,
                "sum",
                new_length as u32,
            );
        }
        Self {
            data: Arc::new(temp_buffer),
            gpu_device: self.get_gpu_device(),
            phantom: std::marker::PhantomData,
            len: 1,
            null_buffer: None,
        }
    }
}
