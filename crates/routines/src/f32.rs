use crate::{merge::U32_MERGE_SHADER, put::U32_PUT_SHADER, take::U32_TAKE_SHADER, SwizzleType};

impl SwizzleType for f32 {
    const MERGE_SHADER: &'static str = U32_MERGE_SHADER;
    const TAKE_SHADER: &'static str = U32_TAKE_SHADER;
    const PUT_SHADER: &'static str = U32_PUT_SHADER;
}

#[cfg(test)]
mod test {
    use crate::*;
    use arrow_gpu_array::array::*;

    test_merge_op!(
        test_merge_f32_array_f32,
        Float32ArrayGPU,
        Float32ArrayGPU,
        Float32ArrayGPU,
        merge,
        merge_dyn,
        [
            Some(0.0),
            Some(1.0),
            None,
            None,
            Some(4.0),
            Some(4.0),
            Some(10.0),
            None,
            Some(50.0)
        ],
        [
            Some(1.0),
            Some(2.0),
            None,
            Some(4.0),
            None,
            None,
            Some(20.0),
            Some(30.0),
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
            Some(0.0),
            Some(1.0),
            None,
            Some(4.0),
            Some(4.0),
            None,
            None,
            None,
            None
        ]
    );

    test_take_op!(
        test_take_f32,
        Float32ArrayGPU,
        UInt32ArrayGPU,
        Float32ArrayGPU,
        take,
        take_dyn,
        [Some(0.0), Some(1.0), None, Some(3.0)],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [
            Some(0.0),
            Some(1.0),
            None,
            Some(3.0),
            Some(0.0),
            Some(1.0),
            None,
            Some(3.0)
        ]
    );

    // TODO test for cases with null
    test_put_op!(
        test_put_f32,
        Float32ArrayGPU,
        put,
        put_dyn,
        [10.0, 1.0, 2.0, 3.0],
        [100.0, 0.0, 101.0, 0.0, 102.0, 0.0, 103.0, 0.0],
        [0, 1, 2, 3],
        [1, 3, 5, 7],
        [100.0, 10.0, 101.0, 1.0, 102.0, 2.0, 103.0, 3.0]
    );
}
