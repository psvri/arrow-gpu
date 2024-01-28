use arrow_gpu_array::array::*;

use crate::{
    u32::{U32_LOGICAL_SHADER, U32_NOT_SHADER},
    *,
};

pub(crate) const ANY_SHADER: &str = include_str!("../compute_shaders/u32/any.wgsl");

impl LogicalType for BooleanArrayGPU {
    const SHADER: &'static str = U32_LOGICAL_SHADER;
    const SHIFT_SHADER: &'static str = "";
    const NOT_SHADER: &'static str = U32_NOT_SHADER;
}

macro_rules! apply_binary_function_op {
    ($self: ident, $operand: ident, $shader: ident, $entry_point: ident, $pipeline: ident) => {
        let dispatch_size = $self.data.size().div_ceil(4).div_ceil(256) as u32;

        let new_buffer = $pipeline.apply_binary_function(
            &$self.data,
            &$operand.data,
            $self.data.size(),
            Self::$shader,
            $entry_point,
            dispatch_size,
        );
        let null_buffer = NullBitBufferGpu::merge_null_bit_buffer_op(
            &$self.null_buffer,
            &$operand.null_buffer,
            $pipeline,
        );

        return Self {
            data: Arc::new(new_buffer),
            gpu_device: $self.gpu_device.clone(),
            len: $self.len,
            null_buffer,
        };
    };
}

impl Logical for BooleanArrayGPU {
    fn bitwise_and_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, AND_ENTRY_POINT, pipeline);
    }

    fn bitwise_or_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, OR_ENTRY_POINT, pipeline);
    }

    fn bitwise_xor_op(&self, operand: &Self, pipeline: &mut ArrowComputePipeline) -> Self {
        apply_binary_function_op!(self, operand, SHADER, XOR_ENTRY_POINT, pipeline);
    }

    fn bitwise_not_op(&self, pipeline: &mut ArrowComputePipeline) -> Self {
        let dispatch_size = self.data.size().div_ceil(4).div_ceil(256) as u32;

        let new_buffer = pipeline.apply_unary_function(
            &self.data,
            self.data.size(),
            Self::NOT_SHADER,
            NOT_ENTRY_POINT,
            dispatch_size,
        );

        Self {
            data: Arc::new(new_buffer),
            gpu_device: self.gpu_device.clone(),
            len: self.len,
            null_buffer: NullBitBufferGpu::clone_null_bit_buffer(&self.null_buffer),
        }
    }

    fn bitwise_shl_op(
        &self,
        operand: &UInt32ArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) -> Self {
        apply_binary_function_op!(
            self,
            operand,
            SHIFT_SHADER,
            SHIFT_LEFT_ENTRY_POINT,
            pipeline
        );
    }

    fn bitwise_shr_op(
        &self,
        operand: &UInt32ArrayGPU,
        pipeline: &mut ArrowComputePipeline,
    ) -> Self {
        apply_binary_function_op!(
            self,
            operand,
            SHIFT_SHADER,
            SHIFT_RIGHT_ENTRY_POINT,
            pipeline
        );
    }
}

impl LogicalContains for BooleanArrayGPU {
    fn any(&self) -> bool {
        let new_buffer = self
            .gpu_device
            .apply_unary_function(&self.data, 4, 4, ANY_SHADER, "any");

        u32::from_le_bytes(
            self.gpu_device
                .retrive_data(&new_buffer)
                .try_into()
                .unwrap(),
        ) > 0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow_gpu_test_macros::{test_array_op, test_unary_op};

    test_array_op!(
        test_bitwise_and_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_and,
        bitwise_and_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None
        ]
    );

    test_array_op!(
        test_bitwise_or_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_or,
        bitwise_or_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![Some(true), Some(true), Some(true), Some(false), None, None]
    );

    test_array_op!(
        test_bitwise_xor_bool_array_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        BooleanArrayGPU,
        bitwise_xor,
        bitwise_xor_dyn,
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            None
        ],
        vec![
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(true)
        ],
        vec![Some(false), Some(true), Some(true), Some(false), None, None]
    );

    test_unary_op!(
        test_bitwise_not_bool,
        BooleanArrayGPU,
        BooleanArrayGPU,
        vec![true, true, false, true, false],
        bitwise_not,
        bitwise_not_dyn,
        vec![false, false, true, false, true]
    );

    #[test]
    fn test_any() {
        use arrow_gpu_array::array::GpuDevice;
        use arrow_gpu_array::GPU_DEVICE;
        let device = GPU_DEVICE
            .get_or_init(|| Arc::new(GpuDevice::new()))
            .clone();
        let data = vec![true, true, false, true, false];
        let gpu_array = BooleanArrayGPU::from_slice(&data, device.clone());
        assert!(gpu_array.any());

        let data = vec![true; 8192 * 2];
        let gpu_array = BooleanArrayGPU::from_slice(&data, device.clone());
        assert!(gpu_array.any());

        let mut data = vec![false; 8192 * 2];
        let gpu_array = BooleanArrayGPU::from_slice(&data, device.clone());
        assert!(!gpu_array.any());

        let mut data_2 = vec![true; 8192 * 2];
        data_2.append(&mut data);
        let gpu_array = BooleanArrayGPU::from_slice(&data_2, device.clone());
        assert!(gpu_array.any());
    }
}
