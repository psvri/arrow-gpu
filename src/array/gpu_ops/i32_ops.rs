use super::div_ceil;
use pollster::FutureExt;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::GpuDevice;

use super::*;

const I32_SCALAR_SHADER: &str = include_str!("../../../compute_shaders/i32/scalar.wgsl");
const I32_ARRAY_SHADER: &str = include_str!("../../../compute_shaders/i32/array.wgsl");
const I32_BROADCAST_SHADER: &str = include_str!("../../../compute_shaders/i32/broadcast.wgsl");

pub async fn add_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, i32, data, value, I32_SCALAR_SHADER, "i32_add");
}

pub async fn sub_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, i32, data, value, I32_SCALAR_SHADER, "i32_sub");
}

pub async fn mul_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, i32, data, value, I32_SCALAR_SHADER, "i32_mul");
}

pub async fn div_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, i32, data, value, I32_SCALAR_SHADER, "i32_div");
}

pub async fn bit_and_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(
        gpu_device,
        i32,
        left,
        right,
        I32_ARRAY_SHADER,
        "bit_and_i32"
    );
}

pub async fn bit_or_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, i32, left, right, I32_ARRAY_SHADER, "bit_or_i32");
}

pub async fn add_array_i32(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, i32, left, right, I32_ARRAY_SHADER, "add_i32");
}

pub fn print_i32_array(gpu_device: &GpuDevice, data: &Buffer, name: &str) {
    let size = data.size() as wgpu::BufferAddress;

    let staging_buffer = gpu_device.create_retrive_buffer(size);
    let mut encoder = gpu_device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_buffer(&data, 0, &staging_buffer, 0, size);

    let submission_index = gpu_device.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    gpu_device
        .device
        .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

    if let Some(Ok(())) = receiver.receive().block_on() {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to i32
        let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        println!("{} {:?}", name, result);
        for i in result {
            println!("{:#032b}", i);
        }
    } else {
        panic!("failed to run compute on gpu!");
    }
}

pub async fn broadcast_i32(gpu_device: &GpuDevice, left: i32, size: u64) -> Buffer {
    broadcast_op!(
        gpu_device,
        i32,
        left,
        I32_BROADCAST_SHADER,
        "broadcast",
        size
    );
}
