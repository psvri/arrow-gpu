use super::div_ceil;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::gpu_array::GpuDevice;

use super::*;

const U32_SCALAR_SHADER: &'static str = include_str!("../../../../compute_shaders/u32_scalar.wgsl");
const U32_ASSIGN_SCALAR_SHADER: &'static str =
    include_str!("../../../../compute_shaders/u32_assign_scalar.wgsl");

pub async fn add_scalar(gpu_device: &GpuDevice, data: &Buffer, value: u32) -> Buffer {
    scalar_op!(gpu_device, u32, data, value, U32_SCALAR_SHADER, "u32_add");
}

pub async fn add_assign_scalar(gpu_device: &GpuDevice, data: &Buffer, value: u32) {
    assign_scalar_op!(
        gpu_device,
        u32,
        data,
        value,
        U32_ASSIGN_SCALAR_SHADER,
        "u32_add_assign"
    );
}
