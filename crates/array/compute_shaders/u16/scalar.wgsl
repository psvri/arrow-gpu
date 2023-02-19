@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read> operand: u32;

@group(0)
@binding(2)
var<storage, write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn u16_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let operand_u16 = get_left_half(operand);

    let left = get_left_half(original_values[global_id.x]) + operand_u16;
    let right = get_right_half(original_values[global_id.x]) + operand_u16;


    new_values[global_id.x] = (left & MAX_U16) + (right << 16u);
}

@compute
@workgroup_size(256)
fn u16_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] - operand;
}

@compute
@workgroup_size(256)
fn u16_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] * operand;
}

@compute
@workgroup_size(256)
fn u16_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] / operand;
}