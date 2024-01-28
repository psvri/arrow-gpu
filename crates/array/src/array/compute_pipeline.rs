use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder};

use super::{gpu_device::CmpQuery, GpuDevice};

pub struct ArrowComputePipeline {
    pub device: Arc<GpuDevice>,
    pub queries: Vec<CmpQuery>,
    pub encoder: CommandEncoder,
}

impl ArrowComputePipeline {
    pub fn new(device: Arc<GpuDevice>, label: Option<&str>) -> Self {
        let encoder = device.create_command_encoder(label);
        Self {
            device,
            queries: vec![],
            encoder,
        }
    }

    pub fn apply_unary_function(
        &mut self,
        original_values: &Buffer,
        new_buffer_size: u64,
        shader: &str,
        entry_point: &str,
        dispatch_size: u32,
    ) -> Buffer {
        let compute_pipeline = self.device.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.device.create_empty_buffer(new_buffer_size);

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

        let query = self.device.compute_pass(
            &mut self.encoder,
            None,
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size,
        );

        query.resolve(&mut self.encoder);

        self.queries.push(query);

        new_values_buffer
    }

    pub fn apply_binary_function(
        &mut self,
        operand_1: &Buffer,
        operand_2: &Buffer,
        new_buffer_size: u64,
        shader: &str,
        entry_point: &str,
        dispatch_size: u32,
    ) -> Buffer {
        let compute_pipeline = self.device.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.device.create_empty_buffer(new_buffer_size);

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

        let query = self.device.compute_pass(
            &mut self.encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size,
        );

        self.queries.push(query);

        new_values_buffer
    }

    pub fn apply_ternary_function(
        &mut self,
        operand_1: &Buffer,
        operand_2: &Buffer,
        operand_3: &Buffer,
        item_size: u64,
        shader: &str,
        entry_point: &str,
    ) -> Buffer {
        let compute_pipeline = self.device.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.device.create_empty_buffer(operand_1.size());

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

        let dispatch_size = operand_1.size() / item_size;

        let query = self.device.compute_pass(
            &mut self.encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size.div_ceil(256) as u32,
        );

        self.queries.push(query);

        new_values_buffer
    }

    pub fn apply_scalar_function(
        &mut self,
        original_values: &Buffer,
        scalar_value: &Buffer,
        output_buffer_size: u64,
        shader: &str,
        entry_point: &str,
        dispatch_size: u32,
    ) -> Buffer {
        let compute_pipeline = self.device.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.device.create_empty_buffer(output_buffer_size);

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

        let query = self.device.compute_pass(
            &mut self.encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size,
        );

        query.resolve(&mut self.encoder);
        self.queries.push(query);

        new_values_buffer
    }

    pub fn apply_broadcast_function(
        &mut self,
        scalar_value: &Buffer,
        output_buffer_size: u64,
        shader: &str,
        entry_point: &str,
        dispatch_size: u32,
    ) -> Buffer {
        let compute_pipeline = self.device.create_compute_pipeline(shader, entry_point);

        let new_values_buffer = self.device.create_empty_buffer(output_buffer_size);

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

        let query = self.device.compute_pass(
            &mut self.encoder,
            Some(entry_point),
            &compute_pipeline,
            &bind_group_array,
            entry_point,
            dispatch_size,
        );

        query.resolve(&mut self.encoder);
        self.queries.push(query);

        new_values_buffer
    }

    pub fn finish(self) {
        let _submision_index = self.device.queue.submit(Some(self.encoder.finish()));

        // There is no point in waiting for the result since while retriving the data we are already blocking it.

        // self.device
        //     .device
        //     .poll(MaintainBase::WaitForSubmissionIndex(submision_index));

        // Find a better way to log profiling data without blocking for it

        // for query in self.queries {
        //     query.wait_for_results(&self.device.device, &self.device.queue);
        // }
    }

    pub fn clone_buffer(&mut self, buffer: &Buffer) -> Buffer {
        let staging_buffer = self.device.create_empty_buffer(buffer.size());

        self.encoder
            .copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        staging_buffer
    }
}
