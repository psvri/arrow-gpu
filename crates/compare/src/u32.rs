use crate::*;

const U32_COMPARE_SHADER: &str = include_str!("../compute_shaders/u32/cmp.wgsl");
const U32_MIN_MAX_SHADER: &str = include_str!("../compute_shaders/u32/min_max.wgsl");

impl CompareType for u32 {
    const COMPARE_SHADER: &'static str = U32_COMPARE_SHADER;
    const MIN_MAX_SHADER: &'static str = U32_MIN_MAX_SHADER;
}

#[cfg(test)]
mod test {
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::test_array_op;

    use crate::*;

    test_array_op!(
        test_gt_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        gt,
        vec![
            Some(0u32),
            Some(3),
            Some(3),
            Some(0),
            Some(3),
            None,
            None,
            Some(4),
            Some(40),
            Some(70)
        ],
        vec![
            Some(1u32),
            Some(2),
            Some(1u32),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        vec![
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            None,
            None,
            None,
            Some(true),
            Some(true)
        ]
    );

    test_array_op!(
        test_all_gt_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        gt,
        vec![Some(100u32); 100],
        vec![Some(1u32); 100],
        vec![Some(true); 100]
    );

    test_array_op!(
        test_gteq_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        gteq,
        vec![
            Some(0u32),
            Some(3),
            Some(3),
            Some(0),
            Some(3),
            None,
            None,
            Some(4),
            Some(40),
            Some(7)
        ],
        vec![
            Some(1u32),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        vec![
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            None,
            None,
            None,
            Some(true),
            Some(true)
        ]
    );

    test_array_op!(
        test_lt_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        lt,
        vec![
            Some(0u32),
            Some(3),
            Some(3),
            Some(0),
            Some(3),
            None,
            None,
            Some(4),
            Some(40),
            Some(7)
        ],
        vec![
            Some(1u32),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        vec![
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            None,
            None,
            None,
            Some(false),
            Some(false)
        ]
    );

    test_array_op!(
        test_lteq_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        lteq,
        vec![
            Some(0u32),
            Some(3),
            Some(3),
            Some(0),
            Some(3),
            None,
            None,
            Some(4),
            Some(40),
            Some(7)
        ],
        vec![
            Some(1u32),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        vec![
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(true),
            None,
            None,
            None,
            Some(false),
            Some(true)
        ]
    );

    test_array_op!(
        test_eq_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        BooleanArrayGPU,
        eq,
        vec![
            Some(0u32),
            Some(3),
            Some(3),
            Some(0),
            Some(3),
            None,
            None,
            Some(4),
            Some(40),
            Some(7)
        ],
        vec![
            Some(1u32),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        vec![
            Some(false),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
            None,
            None,
            None,
            Some(false),
            Some(true)
        ]
    );

    test_array_op!(
        test_min_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        min,
        min_dyn,
        vec![Some(0u32), Some(3), Some(3), Some(0), None, None],
        vec![Some(1u32), Some(2), Some(3), None, Some(4), None],
        vec![Some(0u32), Some(2), Some(3), None, None, None]
    );

    test_array_op!(
        test_max_u32_array_u32,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        UInt32ArrayGPU,
        max,
        max_dyn,
        vec![Some(0u32), Some(3), Some(3), Some(0), None, None],
        vec![Some(1u32), Some(2), Some(3), None, Some(4), None],
        vec![Some(1u32), Some(3), Some(3), None, None, None]
    );
}
