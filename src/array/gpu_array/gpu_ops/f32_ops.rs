use super::div_ceil;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::gpu_array::gpu_ops::assign_scalar_op;
use crate::array::gpu_array::GpuDevice;

use super::scalar_op;

const F32_SCALAR_SHADER: &str = include_str!("../../../../compute_shaders/f32_scalar.wgsl");
const F32_ASSIGN_SCALAR_SHADER: &str =
    include_str!("../../../../compute_shaders/f32_assign_scalar.wgsl");

pub async fn add_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) -> Buffer {
    scalar_op!(gpu_device, f32, data, value, F32_SCALAR_SHADER, "f32_add");
}

pub async fn add_assign_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) {
    assign_scalar_op!(
        gpu_device,
        f32,
        data,
        value,
        F32_ASSIGN_SCALAR_SHADER,
        "f32_add_assign"
    );
}
