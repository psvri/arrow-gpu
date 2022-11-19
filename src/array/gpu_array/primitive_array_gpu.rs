use crate::array::gpu_array::get_device;
use crate::kernels::add_scalar::AddScalarKernel;
use crate::NativeType;
use pollster::FutureExt;
use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::{align_to, DeviceExt};
use wgpu::{Buffer, Device, Queue};

pub struct PrimitiveArrayGpu<T: NativeType> {
    data: Arc<Buffer>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    phantom: PhantomData<T>,
    /// Actual len of the array
    len: usize,
    /// Aligned len of the array since the size of the array in gpu buffer
    /// has to be in multiples of 4bytes
    aligned_len: usize,
}

impl<T: NativeType> PrimitiveArrayGpu<T> {
    pub fn new(mut values: Vec<T>) -> Self {
        let (device, queue) = get_device();
        let element_size = std::mem::size_of::<T>();

        let aligned_size = align_to(values.len() * element_size, 4);
        values.reserve_exact(aligned_size / element_size - values.len());

        let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            data: Arc::new(data),
            device: Arc::new(device),
            queue: Arc::new(queue),
            phantom: Default::default(),
            len: values.len(),
            aligned_len: aligned_size,
        }
    }

    pub fn get_values(&self) -> Option<Vec<T>> {
        let size = self.aligned_len as wgpu::BufferAddress;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.data, 0, &staging_buffer, 0, size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().block_on() {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
            Some(result[0..self.len].to_vec())
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

impl<T: NativeType> Debug for PrimitiveArrayGpu<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\n")?;
        write!(f, "{:?}\n", self.data)?;
        write!(f, "{:?}\n", self.device)?;
        write!(f, "{:?}\n", self.queue)?;
        write!(
            f,
            "Array of length {} contains {:?}\n",
            self.len,
            self.get_values()
        )?;
        write!(f, "}}")
    }
}

macro_rules! add_scalar_primitive {
    ($ty: ident, $shader_path: expr, $group_size: expr) => {
        impl AddScalarKernel<$ty> for PrimitiveArrayGpu<$ty> {
            type Output = PrimitiveArrayGpu<$ty>;

            fn add_scalar(&self, value: $ty) -> Self::Output {
                let cs_module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!($shader_path))),
                    });

                // Gets the size in bytes of the buffer.
                let size = self.aligned_len as wgpu::BufferAddress;

                // Instantiates buffer without data.
                // `usage` of buffer specifies how it can be used:
                //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
                //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
                let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // Instantiates buffer without data.
                // `usage` of buffer specifies how it can be used:
                //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
                //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
                let value_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Value Buffer"),
                            contents: bytemuck::cast_slice(&[value]),
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                // Instantiates the pipeline.
                let compute_pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: None,
                            layout: None,
                            module: &cs_module,
                            entry_point: "main",
                        });

                // Instantiates the bind group, once again specifying the binding of buffers.
                let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
                let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.data.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: value_buffer.as_entire_binding(),
                        },
                    ],
                });

                // A command encoder executes one or many pipelines.
                // It is to WebGPU what a command buffer is to Vulkan.
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group_array, &[]);
                    cpass.insert_debug_marker("compute addition");
                    let dispatch_size = (self.aligned_len / 4) / $group_size;
                    cpass.dispatch_workgroups(dispatch_size as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
                }
                encoder.copy_buffer_to_buffer(&self.data, 0, &staging_buffer, 0, size);

                self.queue.submit(Some(encoder.finish()));

                Self {
                    data: Arc::new(staging_buffer),
                    device: self.device.clone(),
                    queue: self.queue.clone(),
                    phantom: Default::default(),
                    len: self.len,
                    aligned_len: self.aligned_len,
                }
            }
        }
    };
}

add_scalar_primitive!(
    i64,
    "../../../compute_shaders/add_u64_or_i64_scalar.wgsl",
    2
);
add_scalar_primitive!(
    i32,
    "../../../compute_shaders/add_u32_or_i32_scalar.wgsl",
    1
);
add_scalar_primitive!(
    i16,
    "../../../compute_shaders/add_u16_or_i16_scalar.wgsl",
    1
);
add_scalar_primitive!(i8, "../../../compute_shaders/add_u8_or_i8_scalar.wgsl", 1);

add_scalar_primitive!(
    u64,
    "../../../compute_shaders/add_u64_or_i64_scalar.wgsl",
    2
);
add_scalar_primitive!(
    u32,
    "../../../compute_shaders/add_u32_or_i32_scalar.wgsl",
    1
);
add_scalar_primitive!(
    u16,
    "../../../compute_shaders/add_u16_or_i16_scalar.wgsl",
    1
);
add_scalar_primitive!(u8, "../../../compute_shaders/add_u8_or_i8_scalar.wgsl", 1);

add_scalar_primitive!(f32, "../../../compute_shaders/add_f32_scalar.wgsl", 1);

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_add_scalar {
        ($fn_name: ident, $ty: ident, $input: expr, $scalar: expr, $output: expr) => {
            #[test]
            fn $fn_name() {
                let data = $input;
                let gpu_array = PrimitiveArrayGpu::<$ty>::new(data);
                assert_eq!(gpu_array.add_scalar($scalar).get_values().unwrap(), $output)
            }
        };
    }

    test_add_scalar!(
        test_u64_scalar,
        u64,
        vec![0, u64::from(u32::MAX), 18446744069414584320, u64::MAX],
        100,
        vec![100, 4294967395, 18446744069414584420, 99]
    );

    test_add_scalar!(
        test_u32_scalar,
        u32,
        vec![0, 1, 2, 3, 4],
        100,
        vec![100, 101, 102, 103, 104]
    );

    test_add_scalar!(
        test_u16_scalar,
        u16,
        vec![u16::MAX, 1, 2, 3, 4, 0, (u16::MAX / 2)],
        100,
        vec![99, 101, 102, 103, 104, 100, (u16::MAX / 2) + 100]
    );

    test_add_scalar!(
        test_u8_scalar,
        u8,
        vec![0, 1, 2, 3, 4, u8::MAX, (u8::MAX / 2)],
        100,
        vec![100, 101, 102, 103, 104, 99, (u8::MAX / 2) + 100]
    );

    test_add_scalar!(
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

    test_add_scalar!(
        test_i32_scalar,
        i32,
        vec![0, 1, 2, 100, 204],
        -100,
        vec![-100, -99, -98, 0, 104]
    );

    test_add_scalar!(
        test_i16_scalar,
        i16,
        vec![i16::MIN, 1, 2, 100, 204],
        -100,
        vec![i16::MAX - 99, -99, -98, 0, 104]
    );

    test_add_scalar!(
        test_i8_scalar,
        i8,
        vec![0, 1, 2, 100, 125, i8::MAX, i8::MIN],
        -100,
        vec![-100, -99, -98, 0, 25, 27, i8::MAX - 99]
    );

    test_add_scalar!(
        test_f32_scalar,
        f32,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        100.0,
        vec![100.0, 101.0, 102.0, 103.0, 104.0]
    );
}
