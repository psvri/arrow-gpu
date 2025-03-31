use std::sync::Arc;

use arrow_gpu::{
    array::{ArrowArrayGPU, Float32ArrayGPU},
    gpu_utils::{ArrowComputePipeline, GpuDevice},
};
use arrow_gpu_arithmetic::{ArrowScalarAdd, add_scalar_dyn, add_scalar_op_dyn, mul_scalar_op_dyn};

// Basic example of how to create an array on the GPU and run compute kernels
pub fn run_basic_add() {
    // Create a gpu device
    let device = Arc::new(GpuDevice::new());

    let float_values = (0..10).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    // Create a float array on the device
    let gpu_float_array = Float32ArrayGPU::from_slice(&float_values, device.clone());

    let gpu_float_array_scalar = Float32ArrayGPU::from_slice(&[20.0], device.clone());

    // Run an operation on the array
    let add_scalar_result = gpu_float_array.add_scalar(&gpu_float_array_scalar);

    for (index, value) in add_scalar_result.values().iter().enumerate() {
        assert_eq!(value.unwrap(), float_values[index] + 20.0);
    }

    // All the kernels has an equivalent dyn kernel to make life easier which works
    // on enum ArrowArrayGPU
    // For e.g the same same operation can be written as
    let lhs = gpu_float_array.into();
    let rhs = gpu_float_array_scalar.into();
    let dyn_result = add_scalar_dyn(&lhs, &rhs);

    if let ArrowArrayGPU::Float32ArrayGPU(x) = dyn_result {
        for (index, value) in x.values().iter().enumerate() {
            assert_eq!(value.unwrap(), float_values[index] + 20.0);
        }
    } else {
        panic!("Result should be float32 type")
    }
}

/// Each compute kernel can take a compute pipeline,  
/// allowing multiple operations to be executed on the GPU in a single command.
pub fn run_compute_pipeline_ops() {
    let device = Arc::new(GpuDevice::new());

    // Create a compute pipeline which will run on the GPU
    let mut pipeline = ArrowComputePipeline::new(device.clone(), Some("example"));

    let float_values = (0..100).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    // Create float arrays on the device
    let gpu_float_array = Float32ArrayGPU::from_slice(&float_values, device.clone());
    let gpu_float_array_scalar = Float32ArrayGPU::from_slice(&[20.0], device.clone());
    let lhs = gpu_float_array.into();
    let rhs = gpu_float_array_scalar.into();

    // run operations
    let r1 = add_scalar_op_dyn(&lhs, &rhs, &mut pipeline);
    let r2 = mul_scalar_op_dyn(&r1, &rhs, &mut pipeline);

    // call finish on the pipeline to send the commands to the GPU
    pipeline.finish();

    if let ArrowArrayGPU::Float32ArrayGPU(x) = r2 {
        for (index, value) in x.values().iter().enumerate() {
            assert_eq!(value.unwrap(), (float_values[index] + 20.0) * 20.0);
        }
    } else {
        panic!("Result should be float32 type")
    }
}

pub fn main() {
    run_basic_add();
    run_compute_pipeline_ops();
}
