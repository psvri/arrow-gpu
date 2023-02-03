use super::*;
use pollster::FutureExt;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::GpuDevice;

const F32_SCALAR_SHADER: &str = include_str!("../../../compute_shaders/f32_scalar.wgsl");
const F32_ARRAY_SHADER: &str = include_str!("../../../compute_shaders/f32_array.wgsl");
const F32_REDUCTION_SHADER: &str = include_str!("../../../compute_shaders/f32_reduction.wgsl");
const F32_UNARY_SHADER: &str = include_str!("../../../compute_shaders/f32_unary.wgsl");
const F32_BRAODCAST_SHADER: &str = include_str!("../../../compute_shaders/f32/broadcast.wgsl");

pub async fn add_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) -> Buffer {
    scalar_op!(gpu_device, f32, data, value, F32_SCALAR_SHADER, "f32_add");
}

pub async fn sub_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) -> Buffer {
    scalar_op!(gpu_device, f32, data, value, F32_SCALAR_SHADER, "f32_sub");
}

pub async fn mul_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) -> Buffer {
    scalar_op!(gpu_device, f32, data, value, F32_SCALAR_SHADER, "f32_mul");
}

pub async fn div_scalar(gpu_device: &GpuDevice, data: &Buffer, value: f32) -> Buffer {
    scalar_op!(gpu_device, f32, data, value, F32_SCALAR_SHADER, "f32_div");
}

pub async fn add_array_f32(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, u32, left, right, F32_ARRAY_SHADER, "add_f32");
}

pub async fn sum(gpu_device: &GpuDevice, left: &Buffer, mut len: usize) -> f32 {
    //get_f32_array(gpu_device, &left);
    let mut buffer = reduction_op(gpu_device, 4, left, F32_REDUCTION_SHADER, "sum", len);
    //get_f32_array(gpu_device, &buffer);
    len = div_ceil(len as u64, 256) as usize;
    //get_f32_array(gpu_device, &buffer);
    while len > 1 {
        //println!("in loop {:?}", len);
        buffer = reduction_op(gpu_device, 4, &buffer, F32_REDUCTION_SHADER, "sum", len);
        //get_f32_array(gpu_device, &buffer);
        len = div_ceil(len as u64, 256) as usize;
    }

    get_f32_array(gpu_device, &buffer)[0]
}

pub fn get_f32_array(gpu_device: &GpuDevice, data: &Buffer) -> Vec<f32> {
    let data = gpu_device.retrive_data(data).block_on();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    //println!("{:?}", result);
    /*for i in &result {
        println!("{:?}", i);
    }*/
    result
}

pub async fn sin_f32(gpu_device: &GpuDevice, left: &Buffer) -> Buffer {
    unary_op!(gpu_device, u32, left, F32_UNARY_SHADER, "sin_f32");
}

pub async fn braodcast_f32(gpu_device: &GpuDevice, left: f32, size: u64) -> Buffer {
    braodcast_op!(gpu_device, f32, left, F32_BRAODCAST_SHADER, "broadcast", size);
}
