@group(0)
@binding(0)
var<storage, read> left_values: array<u32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn min_(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let left = min(get_left_half(left_values[global_id.x]), get_left_half(right_values[global_id.x]));
    let right = min(get_right_half(left_values[global_id.x]), get_right_half(right_values[global_id.x]));
    new_values[global_id.x] = merge(left, right);
}

@compute
@workgroup_size(256)
fn max_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left = max(get_left_half(left_values[global_id.x]), get_left_half(right_values[global_id.x]));
    let right = max(get_right_half(left_values[global_id.x]), get_right_half(right_values[global_id.x]));
    new_values[global_id.x] = merge(left, right);
}