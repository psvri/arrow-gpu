use crate::{merge::U32_MERGE_SHADER, take::U32_TAKE_SHADER, SwizzleType};

impl SwizzleType for u32 {
    const MERGE_SHADER: &'static str = U32_MERGE_SHADER;
    const TAKE_SHADER: &'static str = U32_TAKE_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;

    test_merge_op!(
        test_merge_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
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

    // TODO test for cases with null
    test_take_op!(
        test_take_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        take,
        vec![0, 1, 2, 3],
        vec![0, 1, 2, 3, 0, 1, 2, 3],
        vec![0, 1, 2, 3, 0, 1, 2, 3]
    );
}
