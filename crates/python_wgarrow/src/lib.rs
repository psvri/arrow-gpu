mod buffer;
mod datatype;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn wgarrow(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    datatype::add_py_items(py, m)?;
    Ok(())
}
