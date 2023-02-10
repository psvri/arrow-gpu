use wgpu::Buffer;
use wgpu::Maintain;

use super::div_ceil;
use super::unary_op;
use crate::array::GpuDevice;

const U8_TRIGONOMETRY_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../../../compute_shaders/u8/trigonometry.wgsl")
);

pub async fn sin_u8(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, u8, left, U8_TRIGONOMETRY_SHADER, "sin_u8", 4);
}
