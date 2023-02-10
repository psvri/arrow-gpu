@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, write> new_values: array<u32>;

@group(0)
@binding(2)
var<storage, read> operand: u32;

@compute
@workgroup_size(256)
fn u32_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] + operand;
}

@compute
@workgroup_size(256)
fn u32_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] - operand;
}

@compute
@workgroup_size(256)
fn u32_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] * operand;
}

@compute
@workgroup_size(256)
fn u32_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] / operand;
}