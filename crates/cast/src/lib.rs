use async_trait::async_trait;

pub(crate) mod boolean_cast;
pub(crate) mod f32_cast;
pub(crate) mod i16_cast;
pub(crate) mod i8_cast;
pub(crate) mod u16_cast;
pub(crate) mod u8_cast;

pub use boolean_cast::*;
pub use f32_cast::*;
pub use i16_cast::*;
pub use i8_cast::*;
pub use u16_cast::*;
pub use u8_cast::*;

use arrow_gpu_array::array::*;

#[async_trait]
pub trait Cast<T> {
    async fn cast(&self) -> T;
}

pub async fn cast_dyn(from: &ArrowArrayGPU, into: &ArrowType) -> ArrowArrayGPU {
    match (from, into) {
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::UInt8Type) => {
            Cast::<UInt8ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::UInt16Type) => {
            Cast::<UInt16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::UInt32Type) => {
            Cast::<UInt32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::Int16Type) => {
            Cast::<Int16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::Int32Type) => {
            Cast::<Int32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int8ArrayGPU(x), ArrowType::Float32Type) => {
            Cast::<Float32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowType::Int32Type) => {
            Cast::<Int32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowType::UInt16Type) => {
            Cast::<UInt16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowType::UInt32Type) => {
            Cast::<UInt32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Int16ArrayGPU(x), ArrowType::Float32Type) => {
            Cast::<Float32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::UInt16Type) => {
            Cast::<UInt16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::UInt32Type) => {
            Cast::<UInt32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::Int8Type) => {
            Cast::<Int8ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::Int16Type) => {
            Cast::<Int16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::Int32Type) => {
            Cast::<Int32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt8ArrayGPU(x), ArrowType::Float32Type) => {
            Cast::<Float32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowType::Int16Type) => {
            Cast::<Int16ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowType::Int32Type) => {
            Cast::<Int32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowType::UInt32Type) => {
            Cast::<UInt32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::UInt16ArrayGPU(x), ArrowType::Float32Type) => {
            Cast::<Float32ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::Float32ArrayGPU(x), ArrowType::UInt8Type) => {
            Cast::<UInt8ArrayGPU>::cast(x).await.into()
        }
        (ArrowArrayGPU::BooleanArrayGPU(x), ArrowType::Float32Type) => {
            Cast::<Float32ArrayGPU>::cast(x).await.into()
        }
        (x, y) => panic!("Casting between {x:?} into {y:?} is not possible"),
    }
}

#[cfg(test)]
mod tests {
    macro_rules! test_cast_op {
        ($(#[$m:meta])* $fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $cast_type: ident, $output: expr) => {
            $(#[$m])*
            #[tokio::test]
            async fn $fn_name() {
                use arrow_gpu_array::GPU_DEVICE;
                let device = GPU_DEVICE.clone();
                let data = $input;
                let gpu_array = $input_ty::from_slice(&data, device);
                let new_gpu_array: $output_ty =
                    <$input_ty as Cast<$output_ty>>::cast(&gpu_array).await;
                let new_values = new_gpu_array.raw_values().await.unwrap();
                assert_eq!(new_values, $output);

                let new_gpu_array = cast_dyn(&gpu_array.into(), &ArrowType::$cast_type).await;
                let new_values = $output_ty::try_from(new_gpu_array)
                    .unwrap()
                    .raw_values()
                    .await
                    .unwrap();
                assert_eq!(new_values, $output);
            }
        };
    }
    pub(crate) use test_cast_op;
}
