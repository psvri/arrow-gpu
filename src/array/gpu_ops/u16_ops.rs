use wgpu::Buffer;
use wgpu::Maintain;

use super::div_ceil;
use super::unary_op;
use crate::array::GpuDevice;

const U16_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../../../compute_shaders/u16/trigonometry.wgsl")
);

pub async fn sin_u16(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, u16, left, U16_TRIGONOMETRY_SHADER, "sin_u16", 2);
}
