use super::Date32Type;

/// Arrow Array backed by i32
pub trait Int32Type {}

impl Int32Type for i32 {}
impl Int32Type for Date32Type {}

/// Arrow Array backed by f32
pub trait Float32Type {}

impl Float32Type for f32 {}

/// Arrow Array backed by u32
pub trait UInt32Type {}

impl UInt32Type for u32 {}

/// Arrow Array backed by u16
pub trait UInt16Type {}

impl UInt16Type for u16 {}
