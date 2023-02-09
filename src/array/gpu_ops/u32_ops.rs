use super::div_ceil;
use pollster::FutureExt;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::GpuDevice;

use super::*;

const U32_SCALAR_SHADER: &str = include_str!("../../../compute_shaders/u32_scalar.wgsl");
const U32_ARRAY_SHADER: &str = include_str!("../../../compute_shaders/u32_array.wgsl");
const U32_BRAODCAST_SHADER: &str = include_str!("../../../compute_shaders/u32/broadcast.wgsl");

pub async fn add_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, u32, data, value, U32_SCALAR_SHADER, "u32_add");
}

pub async fn sub_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, u32, data, value, U32_SCALAR_SHADER, "u32_sub");
}

pub async fn mul_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, u32, data, value, U32_SCALAR_SHADER, "u32_mul");
}

pub async fn div_scalar(gpu_device: &GpuDevice, data: &Buffer, value: &Buffer) -> Buffer {
    scalar_op!(gpu_device, u32, data, value, U32_SCALAR_SHADER, "u32_div");
}

pub async fn bit_and_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(
        gpu_device,
        u32,
        left,
        right,
        U32_ARRAY_SHADER,
        "bit_and_u32"
    );
}

pub async fn bit_or_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, u32, left, right, U32_ARRAY_SHADER, "bit_or_u32");
}

pub async fn add_array_u32(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, u32, left, right, U32_ARRAY_SHADER, "add_u32");
}

pub fn print_u32_array(gpu_device: &GpuDevice, data: &Buffer, name: &str) {
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
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

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

pub async fn braodcast_u32(gpu_device: &GpuDevice, left: u32, size: u64) -> Buffer {
    braodcast_op!(
        gpu_device,
        u32,
        left,
        U32_BRAODCAST_SHADER,
        "broadcast",
        size
    );
}
