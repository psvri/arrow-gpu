use super::*;
use wgpu::Buffer;

use crate::array::gpu_device::GpuDevice;

const F32_REDUCTION_SHADER: &str = include_str!("../../../compute_shaders/f32/reduction.wgsl");

pub fn sum(gpu_device: &GpuDevice, left: &Buffer, mut len: usize) -> f32 {
    //get_f32_array(gpu_device, &left);
    let mut buffer = reduction_op(gpu_device, 4, left, F32_REDUCTION_SHADER, "sum", len);
    //get_f32_array(gpu_device, &buffer);
    len = len.div_ceil(256) as usize;
    //get_f32_array(gpu_device, &buffer);
    while len > 1 {
        //println!("in loop {:?}", len);
        buffer = reduction_op(gpu_device, 4, &buffer, F32_REDUCTION_SHADER, "sum", len);
        //get_f32_array(gpu_device, &buffer);
        len = len.div_ceil(256) as usize;
    }

    get_f32_array(gpu_device, &buffer)[0]
}

pub fn get_f32_array(gpu_device: &GpuDevice, data: &Buffer) -> Vec<f32> {
    let data = gpu_device.retrive_data(data);
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    //println!("{:?}", result);
    /*for i in &result {
        println!("{:?}", i);
    }*/
    result
}
