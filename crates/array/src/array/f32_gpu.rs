use super::{primitive_array_gpu::*, ArrowArray, ArrowArrayGPU, ArrowType, NullBitBufferGpu};
use crate::gpu_utils::*;
use crate::kernels::broadcast::Broadcast;
use crate::ArrowErrorGPU;
use std::{any::Any, sync::Arc};
use wgpu::Buffer;

const F32_BROADCAST_SHADER: &str = include_str!("../../compute_shaders/f32/broadcast.wgsl");

pub type Float32ArrayGPU = PrimitiveArrayGpu<f32>;

impl Broadcast<f32> for Float32ArrayGPU {
    fn broadcast_op(value: f32, len: usize, pipeline: &mut ArrowComputePipeline) -> Self {
        let scalar_buffer = pipeline.device.create_scalar_buffer(&value);
        let output_buffer_size = 4 * len as u64;
        let dispatch_size = output_buffer_size.div_ceil(4).div_ceil(256);

        let gpu_buffer = pipeline.apply_broadcast_function(
            &scalar_buffer,
            output_buffer_size,
            F32_BROADCAST_SHADER,
            "broadcast",
            dispatch_size as u32,
        );
        let data = Arc::new(gpu_buffer);
        let null_buffer = None;

        Self {
            data,
            gpu_device: pipeline.device.clone(),
            phantom: std::marker::PhantomData,
            len,
            null_buffer,
        }
    }
}

impl From<Float32ArrayGPU> for ArrowArrayGPU {
    fn from(val: Float32ArrayGPU) -> Self {
        ArrowArrayGPU::Float32ArrayGPU(val)
    }
}

impl TryFrom<ArrowArrayGPU> for Float32ArrayGPU {
    type Error = ArrowErrorGPU;

    fn try_from(value: ArrowArrayGPU) -> Result<Self, Self::Error> {
        match value {
            ArrowArrayGPU::Float32ArrayGPU(x) => Ok(x),
            x => Err(ArrowErrorGPU::CastingNotSupported(format!(
                "could not cast {:?} into Float32ArrayGPU",
                x
            ))),
        }
    }
}

impl ArrowArray for Float32ArrayGPU {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> ArrowType {
        ArrowType::Float32Type
    }

    fn get_memory_used(&self) -> u64 {
        self.data.size()
    }

    fn get_gpu_device(&self) -> &GpuDevice {
        &self.gpu_device
    }

    fn get_buffer(&self) -> &Buffer {
        &self.data
    }

    fn get_null_bit_buffer(&self) -> Option<&NullBitBufferGpu> {
        self.null_buffer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::primitive_array_gpu::test::*;

    #[test]
    fn test_f32_array_from_optinal_vec() {
        let device = Arc::new(GpuDevice::new());
        let gpu_array_1 = Float32ArrayGPU::from_optional_slice(
            &[Some(0.0), Some(1.0), None, None, Some(4.0)],
            device.clone(),
        );
        assert_eq!(
            gpu_array_1.raw_values().unwrap(),
            vec![0.0, 1.0, 0.0, 0.0, 4.0]
        );
        assert_eq!(
            gpu_array_1.null_buffer.as_ref().unwrap().raw_values(),
            vec![0b00010011]
        );
        let gpu_array_2 = Float32ArrayGPU::from_optional_slice(
            &[Some(1.0), Some(2.0), None, Some(4.0), None],
            device,
        );
        assert_eq!(
            gpu_array_2.raw_values().unwrap(),
            vec![1.0, 2.0, 0.0, 4.0, 0.0]
        );
        assert_eq!(
            gpu_array_2.null_buffer.as_ref().unwrap().raw_values(),
            vec![0b00001011]
        );
        let new_bit_buffer = NullBitBufferGpu::merge_null_bit_buffer(
            &gpu_array_2.null_buffer,
            &gpu_array_1.null_buffer,
        );
        assert_eq!(new_bit_buffer.unwrap().raw_values(), vec![0b00000011]);
    }

    test_broadcast!(test_broadcast_f32, Float32ArrayGPU, 1.0);
}
