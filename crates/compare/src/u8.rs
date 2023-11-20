use crate::*;

const U8_COMPARE_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/cmp.wgsl")
);

//TODO
const U8_MIN_MAX_SHADER: &str = concat!(
    include_str!("../../../compute_shaders/u8/utils.wgsl"),
    include_str!("../compute_shaders/u8/cmp.wgsl")
);

impl CompareType for u8 {
    const COMPARE_SHADER: &'static str = U8_COMPARE_SHADER;
    const MIN_MAX_SHADER: &'static str = U8_MIN_MAX_SHADER;
}

#[cfg(test)]
mod test {
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::test_array_op;

    use crate::*;

    test_array_op!(
        test_gt_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        gt,
        gt_dyn,
        vec![
            Some(0u8),
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
            Some(1u8),
            Some(2),
            Some(1u8),
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
        test_all_gt_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        gt,
        gt_dyn,
        vec![Some(100u8); 100],
        vec![Some(1u8); 100],
        vec![Some(true); 100]
    );

    test_array_op!(
        test_none_gt_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        gt,
        gt_dyn,
        vec![Some(1u8); 100],
        vec![Some(100u8); 100],
        vec![Some(false); 100]
    );

    test_array_op!(
        test_gteq_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        gteq,
        gteq_dyn,
        vec![
            Some(0u8),
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
            Some(1u8),
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
        test_lt_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        lt,
        lt_dyn,
        vec![
            Some(0u8),
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
            Some(1u8),
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
        test_lteq_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        lteq,
        lteq_dyn,
        vec![
            Some(0u8),
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
            Some(1u8),
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
        test_eq_u8_array_u8,
        UInt8ArrayGPU,
        UInt8ArrayGPU,
        BooleanArrayGPU,
        eq,
        eq_dyn,
        vec![
            Some(0u8),
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
            Some(1u8),
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
}
