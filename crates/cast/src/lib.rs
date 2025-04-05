use arrow_gpu_array::array::*;
use arrow_gpu_array::gpu_utils::*;

pub(crate) mod boolean_cast;
pub(crate) mod f32_cast;
pub(crate) mod i16_cast;
pub(crate) mod i8_cast;
pub(crate) mod u16_cast;
pub(crate) mod u32_cast;
pub(crate) mod u8_cast;

pub use boolean_cast::*;

/// The cast ArrowArray into another ArrowArray
pub trait Cast<T>: ArrayUtils {
    fn cast(&self) -> T {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), None);
        let output = self.cast_op(&mut pipeline);
        pipeline.finish();
        output
    }

    /// cast self as `T`
    fn cast_op(&self, pipeline: &mut ArrowComputePipeline) -> T;
}

/// The reinterprets ArrowArray's bits as another ArrowArray
pub trait BitCast<T>: ArrayUtils {
    fn bitcast(&self) -> T {
        let mut pipeline = ArrowComputePipeline::new(self.get_gpu_device(), None);
        let results = self.bitcast_op(&mut pipeline);
        pipeline.finish();
        results
    }

    /// The reinterprets self as `T`
    fn bitcast_op(&self, pipeline: &mut ArrowComputePipeline) -> T;
}

macro_rules! impl_cast {
    ($into_ty: ident, $from_ty: ident, $shader: ident, $entry_point: literal, $item_size: literal, $buffer_size_mul: literal) => {
        impl Cast<$into_ty> for $from_ty {
            fn cast_op(&self, pipeline: &mut ArrowComputePipeline) -> $into_ty {
                let dispatch_size = self.data.size().div_ceil($item_size).div_ceil(256) as u32;

                let new_buffer = pipeline.apply_unary_function(
                    &self.data,
                    self.data.size() * $buffer_size_mul,
                    $shader,
                    $entry_point,
                    dispatch_size,
                );

                let null_buffer = NullBitBufferGpu::clone_null_bit_buffer_pass(
                    &self.null_buffer,
                    &mut pipeline.encoder,
                );

                $into_ty {
                    data: new_buffer.into(),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer,
                }
            }
        }
    };
    ($into_ty: ident, $from_ty: ident) => {
        impl Cast<$into_ty> for $from_ty {
            fn cast_op(&self, pipeline: &mut ArrowComputePipeline) -> $into_ty {
                let new_buffer = pipeline.clone_buffer(&self.data);
                let null_buffer = NullBitBufferGpu::clone_null_bit_buffer_pass(
                    &self.null_buffer,
                    &mut pipeline.encoder,
                );
                $into_ty {
                    data: new_buffer.into(),
                    gpu_device: self.gpu_device.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    null_buffer,
                }
            }
        }
    };
}
pub(crate) use impl_cast;

macro_rules! impl_bitcast {
    ($into_ty: ident, $for_ty: ident) => {
        impl BitCast<$into_ty> for $for_ty {
            fn bitcast_op(&self, pipeline: &mut ArrowComputePipeline) -> $into_ty {
                let data = pipeline.clone_buffer(&self.data);
                let null_buffer =
                    NullBitBufferGpu::clone_null_bit_buffer_op(&self.null_buffer, pipeline);
                let data = data.into();
                $into_ty {
                    data,
                    gpu_device: self.get_gpu_device(),
                    phantom: std::marker::PhantomData,
                    len: self.len,
                    null_buffer,
                }
            }
        }
    };
}
pub(crate) use impl_bitcast;

macro_rules! dyn_cast {
    ($function:ident, $doc: expr, $function_op:ident, $( [$from:ident, $into_ty:ident, $into: ident] ),*) => (
        #[doc=$doc]
        pub fn $function(from: &ArrowArrayGPU, into: &ArrowType) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(from.get_gpu_device(), None);
            let result = $function_op(from, into, &mut pipeline);
            pipeline.finish();
            result
        }

        #[doc=concat!("Submits a command to the pipeline to ", $doc)]
        pub fn $function_op(from: &ArrowArrayGPU, into: &ArrowType, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (from, into) {
                $((ArrowArrayGPU::$from(x), ArrowType::$into_ty) => Cast::<$into>::cast_op(x, pipeline).into(),)+
                _ => panic!(
                    "Casting not supported for type {:?} {:?}",
                    from.get_dtype(),
                    into,
                ),
            }
        }
    )
}

dyn_cast!(
    cast_dyn,
    "Cast x as `T` for each x in array",
    cast_op_dyn,
    [Int8ArrayGPU, UInt8Type, UInt8ArrayGPU],
    [Int8ArrayGPU, UInt16Type, UInt16ArrayGPU],
    [Int8ArrayGPU, UInt32Type, UInt32ArrayGPU],
    [Int8ArrayGPU, Int16Type, Int16ArrayGPU],
    [Int8ArrayGPU, Int32Type, Int32ArrayGPU],
    [Int8ArrayGPU, Float32Type, Float32ArrayGPU],
    [Int16ArrayGPU, Int32Type, Int32ArrayGPU],
    [Int16ArrayGPU, UInt16Type, UInt16ArrayGPU],
    [Int16ArrayGPU, UInt32Type, UInt32ArrayGPU],
    [Int16ArrayGPU, Float32Type, Float32ArrayGPU],
    [UInt8ArrayGPU, UInt16Type, UInt16ArrayGPU],
    [UInt8ArrayGPU, UInt32Type, UInt32ArrayGPU],
    [UInt8ArrayGPU, Int8Type, Int8ArrayGPU],
    [UInt8ArrayGPU, Int16Type, Int16ArrayGPU],
    [UInt8ArrayGPU, Int32Type, Int32ArrayGPU],
    [UInt8ArrayGPU, Float32Type, Float32ArrayGPU],
    [UInt16ArrayGPU, UInt32Type, UInt32ArrayGPU],
    [UInt16ArrayGPU, Int16Type, Int16ArrayGPU],
    [UInt16ArrayGPU, Int32Type, Int32ArrayGPU],
    [UInt16ArrayGPU, Float32Type, Float32ArrayGPU],
    [Float32ArrayGPU, UInt8Type, UInt8ArrayGPU],
    [BooleanArrayGPU, Float32Type, Float32ArrayGPU]
);

macro_rules! dyn_bitcast {
    ($function:ident, $doc: expr, $function_op:ident, $( [$from:ident, $into_ty:ident, $into: ident] ),*) => (
        #[doc=$doc]
        pub fn $function(from: &ArrowArrayGPU, into: &ArrowType) -> ArrowArrayGPU {
            let mut pipeline = ArrowComputePipeline::new(from.get_gpu_device(), None);
            let result = $function_op(from, into, &mut pipeline);
            pipeline.finish();
            result
        }

        #[doc=concat!("Submits a command to the pipeline to ", $doc)]
        pub fn $function_op(from: &ArrowArrayGPU, into: &ArrowType, pipeline: &mut ArrowComputePipeline) -> ArrowArrayGPU {
            match (from, into) {
                $((ArrowArrayGPU::$from(x), ArrowType::$into_ty) => BitCast::<$into>::bitcast_op(x, pipeline).into(),)+
                _ => panic!(
                    "Casting not supported for type {:?} {:?}",
                    from.get_dtype(),
                    into,
                ),
            }
        }
    )
}

dyn_bitcast!(
    bitcast_dyn,
    "Reinterpret x as `T` for each x in array",
    bitcast_op_dyn,
    [UInt32ArrayGPU, Float32Type, Float32ArrayGPU]
);

#[cfg(test)]
mod tests {
    macro_rules! test_cast_op {
        ($(#[$m:meta])* $fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $cast_type: ident, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.clone();
                let gpu_array = $input_ty::from_slice(&$input, device.clone());
                let new_gpu_array: $output_ty =
                    <$input_ty as Cast<$output_ty>>::cast(&gpu_array);
                let new_values = new_gpu_array.raw_values().unwrap();
                assert_eq!(new_values, $output);

                let new_gpu_array = cast_dyn(&gpu_array.into(), &ArrowType::$cast_type);
                let new_values = $output_ty::try_from(new_gpu_array)
                    .unwrap()
                    .raw_values()
                    .unwrap();
                assert_eq!(new_values, $output);
            }
        };
    }
    pub(crate) use test_cast_op;

    macro_rules! test_bitcast_op {
        ($(#[$m:meta])* $fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $cast_type: ident, $output: expr) => {
            $(#[$m])*
            #[test]
            fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.clone();
                let gpu_array = $input_ty::from_slice(&$input, device.clone());
                let new_gpu_array: $output_ty =
                    <$input_ty as BitCast<$output_ty>>::bitcast(&gpu_array);
                let new_values = new_gpu_array.raw_values().unwrap();
                for (index, new_value) in new_values.iter().enumerate() {
                    assert_eq!($output[index].to_bits(), new_value.to_bits());
                }


                let new_gpu_array = bitcast_dyn(&gpu_array.into(), &ArrowType::$cast_type);
                let new_values = $output_ty::try_from(new_gpu_array)
                    .unwrap()
                    .raw_values()
                    .unwrap();
                for (index, new_value) in new_values.iter().enumerate() {
                    assert_eq!($output[index].to_bits(), new_value.to_bits());
                }
            }
        };
    }
    pub(crate) use test_bitcast_op;
}
