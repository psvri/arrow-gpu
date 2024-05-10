use crate::{merge::U32_MERGE_SHADER, put::U32_PUT_SHADER, take::U32_TAKE_SHADER, SwizzleType};

impl SwizzleType for u32 {
    const MERGE_SHADER: &'static str = U32_MERGE_SHADER;
    const TAKE_SHADER: &'static str = U32_TAKE_SHADER;
    const PUT_SHADER: &'static str = U32_PUT_SHADER;
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

    test_take_op!(
        test_take_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        take,
        take_dyn,
        [Some(0), Some(1), None, Some(3)],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [
            Some(0),
            Some(1),
            None,
            Some(3),
            Some(0),
            Some(1),
            None,
            Some(3)
        ]
    );

    // TODO test for cases with null
    test_put_op!(
        test_put_u32,
        UInt32ArrayGPU,
        put,
        put_dyn,
        [0, 1, 2, 3],
        [100, 0, 101, 0, 102, 0, 103, 0],
        [0, 1, 2, 3],
        [1, 3, 5, 7],
        [100, 0, 101, 1, 102, 2, 103, 3]
    );
}
