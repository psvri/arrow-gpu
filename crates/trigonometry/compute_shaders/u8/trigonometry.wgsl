@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<f32>;

@compute
@workgroup_size(256)
fn sin_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let unpacked = unpack4xU8(original_values[global_id.x]);
    let new_pos = global_id.x * 4u;
    new_values[new_pos] = sin(f32(unpacked[0]));
    new_values[new_pos + 1u] = sin(f32(unpacked[1]));
    new_values[new_pos + 2u] = sin(f32(unpacked[2]));
    new_values[new_pos + 3u] = sin(f32(unpacked[3]));
}

@compute
@workgroup_size(256)
fn cos_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let unpacked = unpack4xU8(original_values[global_id.x]);
    let new_pos = global_id.x * 4u;
    new_values[new_pos] = cos(f32(unpacked[0]));
    new_values[new_pos + 1u] = cos(f32(unpacked[1]));
    new_values[new_pos + 2u] = cos(f32(unpacked[2]));
    new_values[new_pos + 3u] = cos(f32(unpacked[3]));
}