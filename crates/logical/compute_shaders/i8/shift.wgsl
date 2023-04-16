@group(0)
@binding(0)
var<storage, read> left_values: array<i32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn bitwise_shl(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index_pos = (global_id.x * 4u);
    let left_byte = get_left_byte(left_values[global_id.x]) << right_values[index_pos];
    let mid_left_byte = get_mid_left_byte(left_values[global_id.x]) << right_values[index_pos + 1u];
    let mid_right_byte = get_mid_right_byte(left_values[global_id.x]) << right_values[index_pos + 2u];
    let right_byte = get_right_byte(left_values[global_id.x]) << right_values[index_pos + 3u];
    new_values[global_id.x] = (left_byte & MAX_I8) | ((mid_left_byte & MAX_I8) << 8u) | ((mid_right_byte & MAX_I8) << 16u) | ((right_byte & MAX_I8) << 24u);
}

@compute
@workgroup_size(256)
fn bitwise_shr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index_pos = (global_id.x * 4u);
    let left_byte = get_left_byte(left_values[global_id.x]) >> right_values[index_pos];
    let mid_left_byte = get_mid_left_byte(left_values[global_id.x]) >> right_values[index_pos + 1u];
    let mid_right_byte = get_mid_right_byte(left_values[global_id.x]) >> right_values[index_pos + 2u];
    let right_byte = get_right_byte(left_values[global_id.x]) >> right_values[index_pos + 3u];
    new_values[global_id.x] = (left_byte & MAX_I8) | ((mid_left_byte & MAX_I8) << 8u) | ((mid_right_byte & MAX_I8) << 16u) | ((right_byte & MAX_I8) << 24u);
}