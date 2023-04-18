use std::borrow::Cow;

use bytemuck::Pod;
use log::info;
use wgpu::{
    util::DeviceExt, Adapter, Buffer, ComputePipeline, Device, Maintain, Queue, ShaderModule,
};

use super::{gpu_ops::div_ceil, RustNativeType};

#[derive(Debug)]
pub struct GpuDevice {
    pub device: Device,
    pub queue: Queue,
}

impl GpuDevice {
    pub async fn new() -> GpuDevice {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        info!("{:?}", device);

        Self { device, queue }
    }

    pub async fn from_adapter(adapter: Adapter) -> GpuDevice {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }

    #[inline]
    pub fn create_shader_module(&self, shader: &str) -> ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
    }

    #[inline]
    pub fn create_compute_pipeline(&self, shader: &str, entry_point: &str) -> ComputePipeline {
        let cs_module = self.create_shader_module(shader);
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point,
            })
    }

    #[inline]
    pub fn create_gpu_buffer_with_data(&self, data: &[impl RustNativeType]) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Values Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    #[inline]
    pub fn create_empty_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_retrive_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[inline]
    pub fn create_scalar_buffer(&self, value: &impl Pod) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalar Buffer"),
                contents: bytemuck::cast_slice(&[*value]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub async fn clone_buffer(&self, buffer: &Buffer) -> Buffer {
        let staging_buffer = self.create_empty_buffer(buffer.size());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        let submission_index = self.queue.submit(Some(encoder.finish()));

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        staging_buffer
    }

    pub async fn retrive_data(&self, data: &Buffer) -> Vec<u8> {
        let size = data.size() as wgpu::BufferAddress;

        let staging_buffer = self.create_retrive_buffer(size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(data, 0, &staging_buffer, 0, size);

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = data.to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // /println!("{:?}", result);

            result
        } else {
            panic!("failed to run compute on gpu!");
        }
    }

    pub async fn apply_unary_function(
        &self,
        original_values: &Buffer,
        new_buffer_size: u64,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.create_empty_buffer(new_buffer_size);

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: original_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_array, &[]);
            cpass.insert_debug_marker(entry_point);
            let dispatch_size = original_values.size() / item_size;
            cpass.dispatch_workgroups(div_ceil(dispatch_size, 256) as u32, 1, 1);
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(Maintain::WaitForSubmissionIndex(submission_index));

        new_values_buffer
    }

    pub async fn apply_scalar_function(
        &self,
        original_values: &Buffer,
        scalar_value: &Buffer,
        output_buffer_size: u64,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.create_empty_buffer(output_buffer_size);

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: original_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scalar_value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(entry_point),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry_point),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_array, &[]);
            cpass.insert_debug_marker(entry_point);
            let dispatch_size = original_values.size() / item_size;
            cpass.dispatch_workgroups(div_ceil(dispatch_size, 256) as u32, 1, 1);
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(Maintain::WaitForSubmissionIndex(submission_index));

        new_values_buffer
    }

    pub async fn apply_binary_function(
        &self,
        operand_1: &Buffer,
        operand_2: &Buffer,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.create_empty_buffer(operand_1.size());

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: operand_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: operand_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(entry_point),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry_point),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_array, &[]);
            cpass.insert_debug_marker(entry_point);
            let dispatch_size = operand_1.size() / item_size;
            cpass.dispatch_workgroups(div_ceil(dispatch_size, 256) as u32, 1, 1);
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(Maintain::WaitForSubmissionIndex(submission_index));

        new_values_buffer
    }

    pub async fn apply_ternary_function(
        &self,
        operand_1: &Buffer,
        operand_2: &Buffer,
        operand_3: &Buffer,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.create_empty_buffer(operand_1.size());

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: operand_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: operand_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: operand_3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(entry_point),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry_point),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_array, &[]);
            cpass.insert_debug_marker(entry_point);
            let dispatch_size = operand_1.size() / item_size;
            println!("{}", dispatch_size);
            cpass.dispatch_workgroups(div_ceil(dispatch_size, 256) as u32, 1, 1);
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(Maintain::WaitForSubmissionIndex(submission_index));

        new_values_buffer
    }

    pub async fn apply_broadcast_function(
        &self,
        scalar_value: &Buffer,
        output_buffer_size: u64,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.create_empty_buffer(output_buffer_size);

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_array = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scalar_value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: new_values_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(entry_point),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry_point),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group_array, &[]);
            cpass.insert_debug_marker(entry_point);
            let dispatch_size = output_buffer_size / item_size;
            cpass.dispatch_workgroups(div_ceil(dispatch_size, 256) as u32, 1, 1);
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(Maintain::WaitForSubmissionIndex(submission_index));

        new_values_buffer
    }
}
