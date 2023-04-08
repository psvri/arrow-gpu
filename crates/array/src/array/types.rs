use super::Date32Type;

pub trait Int32Type {}

impl Int32Type for i32 {}
impl Int32Type for Date32Type {}

pub trait Float32Type {}

impl Float32Type for f32 {}

pub trait UInt32Type {}

impl UInt32Type for u32 {}

pub trait UInt16Type {}

impl UInt16Type for u16 {}
