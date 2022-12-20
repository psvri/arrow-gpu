use super::div_ceil;
use pollster::FutureExt;
use wgpu::Buffer;
use wgpu::Maintain;

use crate::array::gpu_array::GpuDevice;

use super::*;

const U32_SCALAR_SHADER: &str = include_str!("../../../../compute_shaders/u32_scalar.wgsl");
const U32_ASSIGN_SCALAR_SHADER: &str =
    include_str!("../../../../compute_shaders/u32_assign_scalar.wgsl");
const U32_ARRAY_SHADER: &str = include_str!("../../../../compute_shaders/u32_array.wgsl");

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

pub async fn and_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    array_op!(gpu_device, u32, left, right, U32_ARRAY_SHADER, "and_u32");
}

pub async fn add_array(gpu_device: &GpuDevice, left: &Buffer, right: &Buffer) -> Buffer {
    print_u32_array(gpu_device, left);
    print_u32_array(gpu_device, right);
    array_op!(gpu_device, u32, left, right, U32_ARRAY_SHADER, "add_u32");
}

pub fn print_u32_array(gpu_device: &GpuDevice, data: &Buffer) {
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
        println!("inside ops add {:?}", result);
    } else {
        panic!("failed to run compute on gpu!");
    }
}
