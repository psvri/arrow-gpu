#[macro_export]
macro_rules! test_unary_op_float {
    ($fn_name: ident, $input_ty: ident, $output_ty: ident, $input: expr, $unary_fn: ident, $unary_fn_dyn: ident, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            let device = Arc::new(GpuDevice::new().await);
            let data = $input;
            let gpu_array = $input_ty::from_vec(&data, device);
            let new_gpu_array = gpu_array.$unary_fn().await;
            let new_values = new_gpu_array.raw_values().await.unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                        $output, new_values
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
                        "assertion dyn failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                        $output, new_values
                    );
                }
            }
        }
    };
}

pub fn float_eq_in_error(left: f32, right: f32) -> bool {
    !(left == f32::INFINITY && right != f32::INFINITY)
        || (right == f32::INFINITY && left != f32::INFINITY)
        || (right == f32::NEG_INFINITY && left != f32::NEG_INFINITY)
        || (left == f32::NEG_INFINITY && right != f32::NEG_INFINITY)
        || (left.is_nan() && !right.is_nan())
        || (right.is_nan() && !left.is_nan())
        || (left - right) > 0.0002
        || (left - right) < -0.0002
}

#[macro_export]
macro_rules! test_scalar_op {
    ($fn_name: ident, $input_ty: ident, $scalar_ty: ident, $output_ty: ident, $input: expr, $scalar_fn: ident, $scalar_fn_dyn: ident, $scalar: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            let device = Arc::new(GpuDevice::new().await);
            let data = $input;
            let array = $input_ty::from_vec(&data, device.clone());
            let value_array = $scalar_ty::from_vec(&vec![$scalar], device);
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
    ($fn_name: ident, $operand1_type: ident, $operand2_type: ident, $operation: ident, $input_1: expr, $input_2: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            let device = Arc::new(GpuDevice::new().await);
            let gpu_array_1 = $operand1_type::from_optional_vec(&$input_1, device.clone());
            let gpu_array_2 = $operand2_type::from_optional_vec(&$input_2, device);
            let new_gpu_array = gpu_array_1.$operation(&gpu_array_2).await;
            assert_eq!(new_gpu_array.values().await, $output);
        }
    };
}

#[macro_export]
macro_rules! test_float_scalar_op {
    ($fn_name: ident, $input_ty: ident, $scalar_ty: ident, $output_ty: ident, $input: expr, $scalar_fn: ident, $scalar_fn_dyn: ident, $scalar: expr, $output: expr) => {
        #[tokio::test]
        async fn $fn_name() {
            let device = Arc::new(GpuDevice::new().await);
            let data = $input;
            let array = $input_ty::from_vec(&data, device.clone());
            let value_array = $scalar_ty::from_vec(&vec![$scalar], device);
            let new_gpu_array = array.$scalar_fn(&value_array).await;
            let new_values = new_gpu_array.raw_values().await.unwrap();
            for (index, new_value) in new_values.iter().enumerate() {
                if !float_eq_in_error($output[index], *new_value) {
                    panic!(
                        "assertion failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                        $output, new_values
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
                        "assertion dyn failed: `(left == right) \n left: `{:?}` \n right: `{:?}`",
                        $output, new_values
                    );
                }
            }
        }
    };
}
