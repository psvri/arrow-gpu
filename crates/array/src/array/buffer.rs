use std::{ops::Deref, sync::Arc};
use wgpu::Buffer;

#[derive(Debug)]
pub struct ArrowGpuBuffer {
    buffer: Arc<Buffer>,
}

impl ArrowGpuBuffer {
    pub fn to_hex(&self) {
        todo!()
    }

    pub fn equals(&self) {
        todo!()
    }

    pub fn equals_nbytes(&self, _nbytes: u64) {
        todo!()
    }

    pub fn size(&self) -> u64 {
        self.buffer.size()
    }
}

impl AsRef<Buffer> for ArrowGpuBuffer {
    fn as_ref(&self) -> &Buffer {
        &self.buffer
    }
}

impl Deref for ArrowGpuBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl From<Buffer> for ArrowGpuBuffer {
    fn from(value: Buffer) -> Self {
        Self {
            buffer: Arc::new(value),
        }
    }
}

impl From<Arc<Buffer>> for ArrowGpuBuffer {
    fn from(buffer: Arc<Buffer>) -> Self {
        Self { buffer }
    }
}
