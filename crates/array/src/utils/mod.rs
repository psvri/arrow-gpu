#[derive(Debug, PartialEq)]
pub enum ScalarArray {
    F32Vec(Vec<f32>),
    U32Vec(Vec<u32>),
    U16Vec(Vec<u16>),
    U8Vec(Vec<u8>),
    I32Vec(Vec<i32>),
    I16Vec(Vec<i16>),
    I8Vec(Vec<i8>),
    BOOLVec(Vec<bool>),
}

macro_rules! impl_into_scalararray {
    ($from_ty: ident, $into_ty: ident) => {
        impl From<Vec<$from_ty>> for ScalarArray {
            fn from(value: Vec<$from_ty>) -> Self {
                ScalarArray::$into_ty(value)
            }
        }
    };
}

impl_into_scalararray!(f32, F32Vec);
impl_into_scalararray!(u32, U32Vec);
impl_into_scalararray!(u16, U16Vec);
impl_into_scalararray!(u8, U8Vec);
impl_into_scalararray!(i32, I32Vec);
impl_into_scalararray!(i16, I16Vec);
impl_into_scalararray!(i8, I8Vec);
impl_into_scalararray!(bool, BOOLVec);
