use arrow_gpu_array::array::Date32Type;

use crate::*;

const I32_COMPARE_SHADER: &str = include_str!("../compute_shaders/i32/cmp.wgsl");
const I32_MIN_MAX_SHADER: &str = include_str!("../compute_shaders/i32/min_max.wgsl");

impl CompareType for i32 {
    const COMPARE_SHADER: &'static str = I32_COMPARE_SHADER;
    const MIN_MAX_SHADER: &'static str = I32_MIN_MAX_SHADER;
}

impl CompareType for Date32Type {
    const COMPARE_SHADER: &'static str = I32_COMPARE_SHADER;
    const MIN_MAX_SHADER: &'static str = I32_MIN_MAX_SHADER;
}

#[cfg(test)]
mod test {
    use arrow_gpu_array::array::*;
    use arrow_gpu_test_macros::test_array_op;

    use crate::*;

    test_array_op!(
        test_gt_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        BooleanArrayGPU,
        gt,
        vec![
            Some(-1),
            Some(3),
            Some(3),
            Some(-1),
            Some(3),
            None,
            None,
            Some(4),
            Some(400),
            Some(500)
        ],
        vec![
            Some(0),
            Some(2),
            Some(0),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(40),
            Some(50)
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
            Some(true),
        ]
    );

    test_array_op!(
        test_gt_date32_array_date32,
        Date32ArrayGPU,
        Date32ArrayGPU,
        BooleanArrayGPU,
        gt,
        vec![
            Some(-1),
            Some(3),
            Some(3),
            Some(-1),
            Some(3),
            None,
            None,
            Some(4),
            Some(400),
            Some(500)
        ],
        vec![
            Some(0),
            Some(2),
            Some(0),
            Some(2),
            Some(3),
            None,
            Some(4),
            None,
            Some(40),
            Some(50)
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
            Some(true),
        ]
    );

    test_array_op!(
        test_gteq_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        BooleanArrayGPU,
        gteq,
        vec![
            Some(0i32),
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
            Some(1i32),
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
        test_lt_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        BooleanArrayGPU,
        lt,
        vec![
            Some(0i32),
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
            Some(1i32),
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
        test_lteq_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        BooleanArrayGPU,
        lteq,
        vec![
            Some(0i32),
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
            Some(1i32),
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
        test_eq_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        BooleanArrayGPU,
        eq,
        vec![
            Some(0i32),
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
            Some(1i32),
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
        test_min_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        min,
        vec![Some(0i32), Some(3), Some(3), Some(0), None, None],
        vec![Some(1i32), Some(2), Some(3), None, Some(4), None],
        vec![Some(0i32), Some(2), Some(3), None, None, None]
    );

    test_array_op!(
        test_max_i32_array_i32,
        Int32ArrayGPU,
        Int32ArrayGPU,
        Int32ArrayGPU,
        max,
        vec![Some(0i32), Some(3), Some(3), Some(0), None, None],
        vec![Some(1i32), Some(2), Some(3), None, Some(4), None],
        vec![Some(1i32), Some(3), Some(3), None, None, None]
    );
}
