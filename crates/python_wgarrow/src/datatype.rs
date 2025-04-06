use arrow_gpu::array::ArrowType;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "DataType", frozen)]
struct PyArrowDataType {
    data_type: ArrowType,
}

#[pymethods]
impl PyArrowDataType {
    #[getter]
    fn bit_width(&self) -> u32 {
        match self.data_type {
            ArrowType::BooleanType => 1,
            ArrowType::UInt16Type | ArrowType::Int16Type => 16,
            ArrowType::UInt8Type | ArrowType::Int8Type => 8,
            ArrowType::Float32Type
            | ArrowType::UInt32Type
            | ArrowType::Int32Type
            | ArrowType::Date32Type => 32,
            _ => panic!("Unimplemented"),
        }
    }

    #[getter]
    fn byte_width(&self) -> PyResult<u32> {
        match self.data_type {
            ArrowType::UInt16Type | ArrowType::Int16Type => Ok(2),
            ArrowType::UInt8Type | ArrowType::Int8Type => Ok(1),
            ArrowType::Float32Type
            | ArrowType::UInt32Type
            | ArrowType::Int32Type
            | ArrowType::Date32Type => Ok(4),
            ArrowType::BooleanType => Err(PyValueError::new_err("Less than one byte")),
            _ => panic!("Unimplemented"),
        }
    }

    #[getter]
    fn num_fields(&self) -> u32 {
        match self.data_type {
            ArrowType::UInt16Type
            | ArrowType::Int16Type
            | ArrowType::UInt8Type
            | ArrowType::Int8Type
            | ArrowType::Float32Type
            | ArrowType::UInt32Type
            | ArrowType::Int32Type
            | ArrowType::Date32Type
            | ArrowType::BooleanType => 0,
            _ => panic!("Unimplemented"),
        }
    }

    pub fn __repr__(&self) -> String {
        let mut repr = String::from("DataType(");
        repr.push_str(match self.data_type {
            ArrowType::BooleanType => "bool",
            ArrowType::Float32Type => "float",
            ArrowType::UInt32Type => "uint32",
            ArrowType::UInt16Type => "uint16",
            ArrowType::UInt8Type => "uint8",
            ArrowType::Int32Type => "int32",
            ArrowType::Int16Type => "int16",
            ArrowType::Int8Type => "int8",
            ArrowType::Date32Type => todo!(),
            _ => todo!(),
        });
        repr.push(')');

        repr
    }

    pub fn __str__(&self) -> String {
        match self.data_type {
            ArrowType::BooleanType => "bool",
            ArrowType::Float32Type => "float",
            ArrowType::UInt32Type => "uint32",
            ArrowType::UInt16Type => "uint16",
            ArrowType::UInt8Type => "uint8",
            ArrowType::Int32Type => "int32",
            ArrowType::Int16Type => "int16",
            ArrowType::Int8Type => "int8",
            ArrowType::Date32Type => todo!(),
            _ => todo!(),
        }
        .to_string()
    }
}

macro_rules! pydtype_func {
    ($name: ident, $atype: ident) => {
        #[pyfunction]
        fn $name() -> PyArrowDataType {
            PyArrowDataType {
                data_type: ArrowType::$atype,
            }
        }
    };
}

pydtype_func!(_int8, Int8Type);
pydtype_func!(_int16, Int16Type);
pydtype_func!(_int32, Int32Type);
pydtype_func!(_uint8, UInt8Type);
pydtype_func!(_uint16, UInt16Type);
pydtype_func!(_uint32, UInt32Type);
pydtype_func!(_float32, Float32Type);
pydtype_func!(_bool_, BooleanType);

macro_rules! isdtype_func {
    ($name: ident, $atype: pat) => {
        #[pyfunction]
        fn $name(dtype: &Bound<'_, PyArrowDataType>) -> bool {
            matches!(dtype.get().data_type, $atype)
        }
    };
}

isdtype_func!(_is_boolean, ArrowType::BooleanType);

isdtype_func!(_is_int8, ArrowType::Int8Type);
isdtype_func!(_is_int16, ArrowType::Int16Type);
isdtype_func!(_is_int32, ArrowType::Int32Type);

isdtype_func!(_is_uint8, ArrowType::UInt8Type);
isdtype_func!(_is_uint16, ArrowType::UInt16Type);
isdtype_func!(_is_uint32, ArrowType::UInt32Type);

isdtype_func!(_is_float32, ArrowType::Float32Type);

isdtype_func!(
    _is_integer,
    ArrowType::Int8Type
        | ArrowType::Int16Type
        | ArrowType::Int32Type
        | ArrowType::UInt8Type
        | ArrowType::UInt16Type
        | ArrowType::UInt32Type
);

isdtype_func!(
    _is_signed_integer,
    ArrowType::Int8Type | ArrowType::Int16Type | ArrowType::Int32Type
);

isdtype_func!(
    _is_unsigned_integer,
    ArrowType::UInt8Type | ArrowType::UInt16Type | ArrowType::UInt32Type
);

isdtype_func!(_is_floating, ArrowType::Float32Type);

isdtype_func!(
    _is_primitive,
    ArrowType::BooleanType
        | ArrowType::Int8Type
        | ArrowType::Int16Type
        | ArrowType::Int32Type
        | ArrowType::UInt8Type
        | ArrowType::UInt16Type
        | ArrowType::UInt32Type
        | ArrowType::Float32Type
);

pub fn add_py_items(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyArrowDataType>()?;
    m.add_function(wrap_pyfunction!(_int8, m)?)?;
    m.add_function(wrap_pyfunction!(_int16, m)?)?;
    m.add_function(wrap_pyfunction!(_int32, m)?)?;
    m.add_function(wrap_pyfunction!(_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(_float32, m)?)?;
    m.add_function(wrap_pyfunction!(_bool_, m)?)?;

    let types_module = PyModule::new(py, "types")?;
    types_module.add_function(wrap_pyfunction!(_is_boolean, m)?)?;

    types_module.add_function(wrap_pyfunction!(_is_int8, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_int16, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_int32, m)?)?;

    types_module.add_function(wrap_pyfunction!(_is_uint8, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_uint16, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_uint32, m)?)?;

    types_module.add_function(wrap_pyfunction!(_is_float32, m)?)?;

    types_module.add_function(wrap_pyfunction!(_is_integer, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_signed_integer, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_unsigned_integer, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_floating, m)?)?;
    types_module.add_function(wrap_pyfunction!(_is_primitive, m)?)?;

    m.add_submodule(&types_module)?;

    Ok(())
}
