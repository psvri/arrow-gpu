@group(0)
@binding(0)
var<storage, read> left_values: array<f32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<f32>;

@compute
@workgroup_size(256)
fn max_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = max(left_values[global_id.x], right_values[global_id.x]);
}

@compute
@workgroup_size(256)
fn min_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = min(left_values[global_id.x], right_values[global_id.x]);
}