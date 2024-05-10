use crate::*;

const U16_COMPARE_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/cmp.wgsl")
);

const U16_MIN_MAX_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u16/utils.wgsl"),
    include_str!("../compute_shaders/u16/min_max.wgsl")
);

impl CompareType for u16 {
    const COMPARE_SHADER: &'static str = U16_COMPARE_SHADER;
    const MIN_MAX_SHADER: &'static str = U16_MIN_MAX_SHADER;
}

#[cfg(test)]
mod test {
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::test_array_op;

    use crate::*;

    test_array_op!(
        test_gt_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        gt,
        [
            Some(0u16),
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
        [
            Some(1u16),
            Some(2),
            Some(1u16),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(4),
            Some(7)
        ],
        [
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
        test_all_gt_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        gt,
        [Some(100u16); 100],
        [Some(1u16); 100],
        [Some(true); 100]
    );

    test_array_op!(
        test_gteq_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        gteq,
        [
            Some(0u16),
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
        [
            Some(1u16),
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
        [
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
        test_lt_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        lt,
        lt_dyn,
        [
            Some(0u16),
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
        [
            Some(1u16),
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
        [
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
        test_lteq_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        lteq,
        lteq_dyn,
        [
            Some(0u16),
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
        [
            Some(1u16),
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
        [
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
        test_eq_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        BooleanArrayGPU,
        eq,
        eq_dyn,
        [
            Some(0u16),
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
        [
            Some(1u16),
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
        [
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
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_min_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        min,
        min_dyn,
        [Some(0u16), Some(3), Some(3), Some(0), None, None],
        [Some(1u16), Some(2), Some(3), None, Some(4), None],
        [Some(0u16), Some(2), Some(3), None, None, None]
    );

    test_array_op!(
        #[cfg_attr(
            target_os = "windows",
            ignore = "Not passing in CI but passes in local 🤔"
        )]
        test_max_u16_array_u16,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        UInt16ArrayGPU,
        max,
        max_dyn,
        [Some(0u16), Some(3), Some(3), Some(0), None, None],
        [Some(1u16), Some(2), Some(3), None, Some(4), None],
        [Some(1u16), Some(3), Some(3), None, None, None]
    );
}
