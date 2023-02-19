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
