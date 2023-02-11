use wgpu::Buffer;
use wgpu::Maintain;

use super::div_ceil;
use super::unary_op;
use crate::array::GpuDevice;

const I16_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../../../compute_shaders/i16/trigonometry.wgsl")
);

const I16_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i16/utils.wgsl"),
    include_str!("../../../compute_shaders/i16/cast_i32.wgsl")
);

pub async fn sin_i16(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, i16, left, I16_TRIGONOMETRY_SHADER, "sin_i16", 2);
}

pub async fn cast_i32(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, i16, left, I16_CAST_I32_SHADER, "cast_i32", 2);
}
