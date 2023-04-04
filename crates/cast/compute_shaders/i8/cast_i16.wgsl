// functions get_left_half and get_right_half are present in compute_shaders/u16/utils.wgsl
// the rust code concacts them at compile time. This workaround is needed due to lack of import
// support

@group(0)
@binding(0)
var<storage, read> original_values: array<i32>;

@group(0)
@binding(1)
var<storage, write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn cast_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left_byte = get_left_byte(original_values[global_id.x]);
    let mid_left_byte = get_mid_left_byte(original_values[global_id.x]);
    let mid_right_byte = get_mid_right_byte(original_values[global_id.x]);
    let right_byte = get_right_byte(original_values[global_id.x]);
    let new_pos = global_id.x * 2u;
    new_values[new_pos] = ((left_byte & 0x0000ffff) | (mid_left_byte << 16u));
    new_values[new_pos + 1u] = ((mid_right_byte & 0x0000ffff) | (right_byte << 16u));
}