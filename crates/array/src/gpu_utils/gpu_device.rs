use std::{borrow::Cow, fmt::Debug, sync::Arc};

use bytemuck::Pod;
use hashbrown::Equivalent;
use log::info;
use pollster::FutureExt;
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, Buffer, ComputePipeline, Device, MemoryHints, Queue,
    ShaderModule, util::DeviceExt,
};

use crate::array::RustNativeType;

use super::{CmpQuery, append_hashmap::AppendHashMap};

#[derive(PartialEq, Eq, Hash)]
struct PiepelineEntry {
    shader: String,
    entry_point: String,
}

impl Equivalent<PiepelineEntry> for (&str, &str) {
    fn equivalent(&self, key: &PiepelineEntry) -> bool {
        self.0 == key.shader && self.1 == key.entry_point
    }
}

pub struct GpuDevice {
    pub device: Device,
    pub queue: Queue,
    pipeline_cache: AppendHashMap<PiepelineEntry, ComputePipeline>,
}

impl Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish()
    }
}

impl GpuDevice {
    pub fn new() -> GpuDevice {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .unwrap();

        let required_features = if cfg!(feature = "profile") {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .unwrap();

        info!("{:?}", device);

        Self {
            device,
            queue,
            pipeline_cache: AppendHashMap::new(),
        }
    }

    pub fn from_adapter(adapter: Adapter) -> GpuDevice {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .block_on()
            .unwrap();

        Self {
            device,
            queue,
            pipeline_cache: AppendHashMap::new(),
        }
    }

    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    pub fn create_query_set(&self) -> CmpQuery {
        CmpQuery::new(&self.device)
    }

    pub fn compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: Option<&str>,
        compute_pipeline: &ComputePipeline,
        bind_group_array: &BindGroup,
        entry_point: &str,
        dispatch_size: u32,
    ) -> CmpQuery {
        let query = self.create_query_set();
        let compute_pass_descriptor = query.create_compute_pass_descriptor(label);
        let mut cpass = encoder.begin_compute_pass(&compute_pass_descriptor);
        cpass.set_pipeline(compute_pipeline);
        cpass.set_bind_group(0, bind_group_array, &[]);
        cpass.insert_debug_marker(entry_point);
        cpass.dispatch_workgroups(dispatch_size, 1, 1);
        query
    }

    pub fn create_shader_module(&self, shader: &str) -> ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
    }

    pub fn create_compute_pipeline(&self, shader: &str, entry_point: &str) -> Arc<ComputePipeline> {
        if let Some(pipeline) = self.pipeline_cache.get(&(shader, entry_point)) {
            pipeline
        } else {
            let cs_module = self.create_shader_module(shader);
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &cs_module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
            self.pipeline_cache.insert(
                PiepelineEntry {
                    shader: shader.into(),
                    entry_point: entry_point.into(),
                },
                pipeline,
            )
        }
    }

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

    pub fn create_retrive_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn create_scalar_buffer(&self, value: &impl Pod) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scalar Buffer"),
                contents: bytemuck::cast_slice(&[*value]),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn clone_buffer(&self, buffer: &Buffer) -> Buffer {
        let staging_buffer = self.create_empty_buffer(buffer.size());

        let mut encoder = self.create_command_encoder(None);

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        self.queue.submit(Some(encoder.finish()));

        staging_buffer
    }

    pub fn clone_buffer_pass(&self, buffer: &Buffer, encoder: &mut wgpu::CommandEncoder) -> Buffer {
        let staging_buffer = self.create_empty_buffer(buffer.size());

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        staging_buffer
    }

    pub fn retrive_data(&self, data: &Buffer) -> Vec<u8> {
        let size = data.size() as wgpu::BufferAddress;

        let staging_buffer = self.create_retrive_buffer(size);
        let mut encoder = self.create_command_encoder(None);

        encoder.copy_buffer_to_buffer(data, 0, &staging_buffer, 0, size);

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        if let Some(Ok(())) = receiver.receive().block_on() {
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

    pub fn apply_unary_function(
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

        let mut encoder = self.create_command_encoder(None);
        let dispatch_size = original_values.size().div_ceil(item_size);

        let query = self.compute_pass(
            &mut encoder,
            None,
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        query.resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        new_values_buffer
    }

    pub fn apply_scalar_function(
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

        let mut encoder = self.create_command_encoder(Some(entry_point));
        let dispatch_size = original_values.size() / item_size;

        self.compute_pass(
            &mut encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        self.queue.submit(Some(encoder.finish()));

        new_values_buffer
    }

    pub fn apply_binary_function(
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

        let mut encoder = self.create_command_encoder(Some(entry_point));
        let dispatch_size = operand_1.size() / item_size;

        self.compute_pass(
            &mut encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        self.queue.submit(Some(encoder.finish()));

        new_values_buffer
    }

    pub fn apply_ternary_function(
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

        let mut encoder = self.create_command_encoder(Some(entry_point));
        let dispatch_size = operand_1.size() / item_size;

        self.compute_pass(
            &mut encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        self.queue.submit(Some(encoder.finish()));

        new_values_buffer
    }

    pub fn apply_broadcast_function(
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

        let mut encoder = self.create_command_encoder(Some(entry_point));
        let dispatch_size = output_buffer_size / item_size;

        self.compute_pass(
            &mut encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        self.queue.submit(Some(encoder.finish()));

        new_values_buffer
    }

    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        self.device.create_bind_group(desc)
    }
}
