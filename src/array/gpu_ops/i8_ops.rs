use wgpu::Buffer;
use wgpu::Maintain;

use super::div_ceil;
use super::unary_op;
use crate::array::GpuDevice;

const I8_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../../../compute_shaders/i8/trigonometry.wgsl")
);

const I8_CAST_I32_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/i8/utils.wgsl"),
    include_str!("../../../compute_shaders/i8/cast_i32.wgsl")
);

pub async fn sin_i8(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, i8, left, I8_TRIGONOMETRY_SHADER, "sin_i8", 4);
}

pub async fn cast_i32(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, i8, left, I8_CAST_I32_SHADER, "cast_i32", 4);
}
