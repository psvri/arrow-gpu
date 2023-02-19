pub mod f32_ops;

use super::gpu_device::GpuDevice;
use wgpu::{util::align_to, Buffer, Maintain};

pub(crate) fn div_ceil(x: u64, y: u64) -> u64 {
    x / y + ((x % y > 0) as u64)
}

pub(crate) fn reduction_op(
    gpu_device: &GpuDevice,
    item_size: usize,
    data: &Buffer,
    shader: &'static str,
    entry_point: &str,
    len: usize,
) -> Buffer {
    let compute_pipeline = gpu_device.create_compute_pipeline(shader, entry_point);

    let size = data.size();
    //println!("input size is {:?}", size);
    let value_buffer = gpu_device.create_scalar_buffer(&len);
    let new_values_buffer = gpu_device.create_empty_buffer(align_to(div_ceil(size, 256), 4));
    /*println!(
        "new_values_buffer size is {:?} {:?}",
        new_values_buffer.size(),
        div_ceil(size as u64, 256)
    );*/

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group_array = gpu_device
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = gpu_device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group_array, &[]);
        cpass.insert_debug_marker("add array");
        let dispatch_size = new_values_buffer.size() / item_size as u64;
        //println!("dispathch size is {:?}", dispatch_size);
        cpass.dispatch_workgroups(dispatch_size as u32, 1, 1);
    }

    let submission_index = gpu_device.queue.submit(Some(encoder.finish()));
    gpu_device
        .device
        .poll(Maintain::WaitForSubmissionIndex(submission_index));

    new_values_buffer
}

/*pub fn get_alignment(len: u64, item_size: u64) -> u64 {
    match item_size {
        1 => align_to(div_ceil(len, item_size), 256),
        2 => align_to(div_ceil(len, item_size), 256),
        _ => align_to(len, 256),
    }
}

add_assign_primitive!(
    i64,
    "../../../compute_shaders/add_u64_or_i64_scalar.wgsl",
    2
);
add_assign_primitive!(
    i32,
    "../../../compute_shaders/add_u32_or_i32_scalar.wgsl",
    1
);
add_assign_primitive!(
    i16,
    "../../../compute_shaders/add_u16_or_i16_scalar.wgsl",
    1
);
add_assign_primitive!(i8, "../../../compute_shaders/add_u8_or_i8_scalar.wgsl", 1);

add_assign_primitive!(
    u64,
    "../../../compute_shaders/add_u64_or_i64_scalar.wgsl",
    2
);
add_assign_primitive!(
    u16,
    "../../../compute_shaders/add_u16_or_i16_scalar.wgsl",
    1
);
add_assign_primitive!(u8, "../../../compute_shaders/add_u8_or_i8_scalar.wgsl", 1);
*/

/*
test_add_assign_scalar!(
        test_u64_scalar,
        u64,
        vec![0, u64::from(u32::MAX), 18446744069414584320, u64::MAX],
        100,
        vec![100, 4294967395, 18446744069414584420, 99]
    );

    test_add_assign_scalar!(
        test_u16_scalar,
        u16,
        vec![u16::MAX, 1, 2, 3, 4, 0, (u16::MAX / 2)],
        100,
        vec![99, 101, 102, 103, 104, 100, (u16::MAX / 2) + 100]
    );

    test_add_assign_scalar!(
        test_u8_scalar,
        u8,
        vec![0, 1, 2, 3, 4, u8::MAX, (u8::MAX / 2)],
        100,
        vec![100, 101, 102, 103, 104, 99, (u8::MAX / 2) + 100]
    );

    test_add_assign_scalar!(
        test_i64_scalar,
        i64,
        vec![4294967296, -4294967296, 2, 100, 204, i64::MAX, i64::MIN],
        -100,
        vec![
            4294967196,
            -4294967396,
            -98,
            0,
            104,
            i64::MAX - 100,
            i64::MAX - 99
        ]
    );

    test_add_assign_scalar!(
        test_i32_scalar,
        i32,
        vec![0, 1, 2, 100, 204],
        -100,
        vec![-100, -99, -98, 0, 104]
    );

    test_add_assign_scalar!(
        test_i16_scalar,
        i16,
        vec![i16::MIN, 1, 2, 100, 204],
        -100,
        vec![i16::MAX - 99, -99, -98, 0, 104]
    );

    test_add_assign_scalar!(
        test_i8_scalar,
        i8,
        vec![0, 1, 2, 100, 125, i8::MAX, i8::MIN],
        -100,
        vec![-100, -99, -98, 0, 25, 27, i8::MAX - 99]
    );
*/
