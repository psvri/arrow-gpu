@group(0)
@binding(0)
var<storage, read> original_values: array<i32>;

@group(0)
@binding(1)
var<storage, read> operand: i32;

@group(0)
@binding(2)
var<storage, write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn i32_rem(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] % operand;
}


@compute
@workgroup_size(256)
fn i32_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] + operand;
}

@compute
@workgroup_size(256)
fn i32_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] - operand;
}

@compute
@workgroup_size(256)
fn i32_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] * operand;
}

@compute
@workgroup_size(256)
fn i32_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] / operand;
}