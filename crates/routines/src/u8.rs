use crate::{merge::U8_MERGE_SHADER, SwizzleType};

impl SwizzleType for u8 {
    const MERGE_SHADER: &'static str = U8_MERGE_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::*;
    use std::sync::Arc;

    test_merge_op!(
        test_merge_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        merge,
        merge_dyn,
        vec![
            Some(0),
            Some(1),
            None,
            None,
            Some(4),
            Some(4),
            Some(10),
            None,
            Some(50)
        ],
        vec![
            Some(1),
            Some(2),
            None,
            Some(4),
            None,
            None,
            Some(20),
            Some(30),
            None
        ],
        vec![
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            None,
            None,
            Some(false),
        ],
        vec![
            Some(0),
            Some(1),
            None,
            Some(4),
            Some(4),
            None,
            None,
            None,
            None
        ]
    );
}
