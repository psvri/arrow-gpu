@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, write> new_values: array<f32>;


@compute
@workgroup_size(256)
fn sinh_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left_byte = get_left_byte(original_values[global_id.x]);
    let mid_left_byte = get_mid_left_byte(original_values[global_id.x]);
    let mid_right_byte = get_mid_right_byte(original_values[global_id.x]);
    let right_byte = get_right_byte(original_values[global_id.x]);
    let new_pos = global_id.x * 4u;
    new_values[new_pos] = sinh(f32(left_byte));
    new_values[new_pos + 1u] = sinh(f32(mid_left_byte));
    new_values[new_pos + 2u] = sinh(f32(mid_right_byte));
    new_values[new_pos + 3u] = sinh(f32(right_byte));
}