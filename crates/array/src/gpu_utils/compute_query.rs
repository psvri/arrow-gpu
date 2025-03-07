use wgpu::ComputePassDescriptor;

#[cfg(feature = "profile")]
const NUM_QUERIES: u64 = 2;

#[cfg(feature = "profile")]
pub struct CmpQuery {
    pub set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
}

#[cfg(feature = "profile")]
impl CmpQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        CmpQuery {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: NUM_QUERIES as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: std::mem::size_of::<u64>() as u64 * NUM_QUERIES,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: std::mem::size_of::<u64>() as u64 * NUM_QUERIES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            // TODO(https://github.com/gfx-rs/wgpu/issues/3993): Musn't be larger than the number valid queries in the set.
            0..2,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn wait_for_results(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait);

        let period = queue.get_timestamp_period();
        let timestamps: Vec<u64> = {
            let timestamp_view = self
                .destination_buffer
                .slice(..(std::mem::size_of::<u64>() as wgpu::BufferAddress * NUM_QUERIES))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        log::info!(
            "Time taken for compute pass is : {:?} ms",
            timestamps[1].wrapping_sub(timestamps[0]) as f64 * period as f64 / 1000.0
        );
    }

    pub fn create_compute_pass_descriptor<'a>(
        &'a self,
        label: Option<&'a str>,
    ) -> wgpu::ComputePassDescriptor<'a> {
        wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        }
    }
}

#[cfg(not(feature = "profile"))]
pub struct CmpQuery {}

#[cfg(not(feature = "profile"))]
impl CmpQuery {
    pub fn new(_device: &wgpu::Device) -> Self {
        Self {}
    }

    pub fn resolve(&self, _encoder: &mut wgpu::CommandEncoder) {}

    pub fn create_compute_pass_descriptor<'a>(
        &'a self,
        label: Option<&'a str>,
    ) -> ComputePassDescriptor<'a> {
        ComputePassDescriptor {
            label,
            timestamp_writes: None,
        }
    }

    pub fn wait_for_results(&self, _device: &wgpu::Device, _queue: &wgpu::Queue) {}
}
