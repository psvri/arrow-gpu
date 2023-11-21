#[macro_export]
macro_rules! test_unary_op {
    ($fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $unary_fn: ident, $unary_fn_dyn: ident, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let data = $input;
            let gpu_array = $input_ty::from_slice(&data, device);
            let new_gpu_array = gpu_array.$unary_fn().await;
            assert_eq!(new_gpu_array.raw_values().await.unwrap(), $output);
            let new_gpu_array = $unary_fn_dyn(&gpu_array.into()).await;
            let new_values = $output_ty::try_from(new_gpu_array)
                .unwrap()
                .raw_values()
                .await
                .unwrap();
            assert_eq!(new_values, $output);
        }
    };
    ($fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $unary_fn: ident, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let data = $input;
            let gpu_array = $input_ty::from_slice(&data, device);
            let new_gpu_array = gpu_array.$unary_fn().await;
            assert_eq!(new_gpu_array.raw_values().await.unwrap(), $output);
        }
    };
}

#[macro_export]
macro_rules! test_scalar_op {
    ($fn_name: ident, $input_ty: ident, $scalar_ty: ident, $output_ty: ident, $input: expr, $scalar_fn: ident, $scalar_fn_dyn: ident, $scalar: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let data = $input;
            let array = $input_ty::from_slice(&data, device.clone());
            let value_array = $scalar_ty::from_slice(&vec![$scalar], device);
            let new_array = array.$scalar_fn(&value_array).await;
            assert_eq!(new_array.raw_values().await.unwrap(), $output);

            let new_gpu_array = $scalar_fn_dyn(&array.into(), &value_array.into()).await;
            let new_values = $output_ty::try_from(new_gpu_array)
                .unwrap()
                .raw_values()
                .await
                .unwrap();
            assert_eq!(new_values, $output);
        }
    };
}

#[macro_export]
macro_rules! test_array_op {
    ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $output: expr) => {
        $(#[$m])*
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
            let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device);
            let new_gpu_array = gpu_array_1.$operation(&gpu_array_2).await;
            assert_eq!(new_gpu_array.values().await, $output);
        }
    };
    ($(#[$m:meta])* $fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr, $output: expr) => {
        $(#[$m])*
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
            let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device);
            let new_gpu_array = gpu_array_1.$operation(&gpu_array_2).await;
            assert_eq!(new_gpu_array.values().await, $output);

            let new_gpu_array = $operation_dyn(&gpu_array_1.into(), &gpu_array_2.into()).await;
            let new_values = $output_type::try_from(new_gpu_array)
                .unwrap()
                .values()
                .await;
            assert_eq!(new_values, $output);
        }
    };
}

pub fn float_eq_in_error(left: f32, right: f32) -> bool {
    if (left.is_nan() && !right.is_nan()) || (right.is_nan() && !left.is_nan()) {
        return false;
    }
    if left.is_nan() && right.is_nan() {
        return true;
    }
    if (right == f32::NEG_INFINITY && left != f32::NEG_INFINITY)
        || (left == f32::NEG_INFINITY && right != f32::NEG_INFINITY)
    {
        return false;
    }
    if (left == f32::INFINITY && right != f32::INFINITY)
        || (right == f32::INFINITY && left != f32::INFINITY)
    {
        return false;
    }
    if (left.abs() - right.abs()).abs() > 0.01 {
        return false;
    }
    true
}

pub fn float_eq_in_error_optional(left: Option<f32>, right: Option<f32>) -> bool {
    match (left, right) {
        (Some(x), Some(y)) => float_eq_in_error(x, y),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

#[macro_export]
macro_rules! test_float_scalar_op {
    ($fn_name: ident, $input_ty: ident, $scalar_ty: ident, $output_ty: ident, $input: expr, $scalar_fn: ident, $scalar_fn_dyn: ident, $scalar: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let data = $input;
            let array = $input_ty::from_slice(&data, device.clone());
            let value_array = $scalar_ty::from_slice(&vec![$scalar], device);
            let new_gpu_array = array.$scalar_fn(&value_array).await;
            let new_values = new_gpu_array.raw_values().await.unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }

            let new_gpu_array = $scalar_fn_dyn(&array.into(), &value_array.into()).await;
            let new_values = $output_ty::try_from(new_gpu_array)
                .unwrap()
                .raw_values()
                .await
                .unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion dyn failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_unary_op_float {
    ($(#[$m:meta])* $fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $unary_fn: ident, $unary_fn_dyn: ident, $output: expr) => {
        $(#[$m])*
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let data = $input;
            let gpu_array = $input_ty::from_slice(&data, device);
            let new_gpu_array = gpu_array.$unary_fn().await;
            let new_values = new_gpu_array.raw_values().await.unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }

            let new_gpu_array = $unary_fn_dyn(&(gpu_array.into())).await;
            let new_values = $output_ty::try_from(new_gpu_array)
                .unwrap()
                .raw_values()
                .await
                .unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion dyn failed: `(left {} == right {}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_float_array_op {
    ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
            let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device);
            let new_gpu_array = gpu_array_1.$operation(&gpu_array_2).await;
            let new_values = new_gpu_array.values().await;
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error_optional($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left {:?} == right {:?}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }
        }
    };
    ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $output_type: ident, $operation: ident, $operation_dyn: ident, $input_1: expr, $input_2: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            use arrow_gpu_array::GPU_DEVICE;
            let device = GPU_DEVICE.clone();
            let gpu_array_1 = $operand1_type::from_optional_slice(&$input_1, device.clone());
            let gpu_array_2 = $operand2_type::from_optional_slice(&$input_2, device);
            let new_gpu_array = gpu_array_1.$operation(&gpu_array_2).await;
            let new_values = new_gpu_array.values().await;
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error_optional($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left {:?} == right {:?}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }

            let new_gpu_array = $operation_dyn(&gpu_array_1.into(), &gpu_array_2.into()).await;
            let new_values = $output_type::try_from(new_gpu_array)
                .unwrap()
                .values()
                .await;
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error_optional($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left {:?} == right {:?}) \n left: `{:?}` \n right: `{:?}`",
                        $output[index], *new_value, $output, new_values
                    );
                }
            }
        }
    };
}

#[cfg(test)]
mod test {
    use crate::float_eq_in_error_optional;

    #[test]
    fn test_float_eq_in_error_optional() {
        // test nan cases
        assert!(float_eq_in_error_optional(Some(f32::NAN), Some(f32::NAN)));
        assert!(!float_eq_in_error_optional(Some(f32::NAN), Some(0.0)));
        assert!(!float_eq_in_error_optional(Some(0.0), Some(f32::NAN)));

        // test inf cases
        assert!(float_eq_in_error_optional(
            Some(f32::INFINITY),
            Some(f32::INFINITY)
        ));
        assert!(float_eq_in_error_optional(
            Some(f32::NEG_INFINITY),
            Some(f32::NEG_INFINITY)
        ));
        assert!(!float_eq_in_error_optional(
            Some(f32::NEG_INFINITY),
            Some(f32::INFINITY)
        ));
        assert!(!float_eq_in_error_optional(Some(f32::INFINITY), Some(0.0)));
        assert!(!float_eq_in_error_optional(
            Some(f32::INFINITY),
            Some(f32::NAN)
        ));
        assert!(!float_eq_in_error_optional(
            Some(f32::NEG_INFINITY),
            Some(0.0)
        ));
        assert!(!float_eq_in_error_optional(
            Some(f32::NEG_INFINITY),
            Some(f32::NAN)
        ));

        assert!(float_eq_in_error_optional(Some(0.0), Some(0.0)));
        assert!(!float_eq_in_error_optional(Some(1.0), Some(0.0)));
        assert!(float_eq_in_error_optional(Some(0.00003), Some(0.0)));
    }
}
