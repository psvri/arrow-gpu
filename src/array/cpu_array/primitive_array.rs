use crate::NativeType;
use std::sync::Arc;

pub struct PrimitiveArray<T: NativeType> {
    data: Arc<Vec<T>>,
}

impl<T: NativeType> PrimitiveArray<T> {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Arc::new(Vec::with_capacity(capacity)),
        }
    }
}

impl<T: NativeType> From<Vec<T>> for PrimitiveArray<T> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            data: Arc::new(vec),
        }
    }
}
