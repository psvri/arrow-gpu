use arrow_gpu::array::buffer::ArrowGpuBuffer;
use pyo3::prelude::*;

#[pyclass(name = "Buffer")]
struct PyBuffer {
    buffer: ArrowGpuBuffer,
}

#[pymethods]
impl PyBuffer {}
