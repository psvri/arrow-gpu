@group(0)
@binding(0)
var<storage, read> original_values: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<f32>;


@compute
@workgroup_size(256)
fn sqrt_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = sqrt(original_values[global_id.x]);
}

@compute
@workgroup_size(256)
fn exp_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = exp(original_values[global_id.x]);
}

@compute
@workgroup_size(256)
fn exp2_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = exp2(original_values[global_id.x]);
}

@compute
@workgroup_size(256)
fn log_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = log(original_values[global_id.x]);
}

@compute
@workgroup_size(256)
fn log2_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = log2(original_values[global_id.x]);
}