@group(0)
@binding(0)
var<storage, read> original_values: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn cast_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = 4u * global_id.x;
    let left_byte = u32(original_values[index]) % 256u;
    let mid_left_byte = u32(original_values[index + 1u]) % 256u;
    let mid_right_byte = u32(original_values[index + 2u]) % 256u;
    let right_byte = u32(original_values[index + 3u]) % 256u;
    new_values[global_id.x] = new_values[global_id.x] | left_byte;
    new_values[global_id.x] = new_values[global_id.x] | (mid_left_byte << 8u);
    new_values[global_id.x] = new_values[global_id.x] | (mid_right_byte << 16u);
    new_values[global_id.x] = new_values[global_id.x] | (right_byte << 24u);
}