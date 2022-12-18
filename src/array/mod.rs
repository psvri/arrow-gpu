use bytemuck::Pod;
use std::fmt::Debug;

pub mod gpu_array;

pub trait NativeType: Pod + Debug {}

impl NativeType for f32 {}
impl NativeType for i64 {}
impl NativeType for i32 {}
impl NativeType for i16 {}
impl NativeType for i8 {}
impl NativeType for u64 {}
impl NativeType for u32 {}
impl NativeType for u16 {}
impl NativeType for u8 {}


