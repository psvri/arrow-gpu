use crate::{SwizzleType, merge::U8_MERGE_SHADER};

impl SwizzleType for u8 {
    const MERGE_SHADER: &'static str = U8_MERGE_SHADER;
    const TAKE_SHADER: &'static str = "todo!()";
    const PUT_SHADER: &'static str = "todo!()";
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::*;

    test_merge_op!(
        test_merge_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        merge,
        merge_dyn,
        [
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
        [
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
        [
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
        [
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
