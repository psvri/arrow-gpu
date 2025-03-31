@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<f32>;


@compute
@workgroup_size(256)
fn sinh_u8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let unpacked = unpack4xU8(original_values[global_id.x]);
    let new_pos = global_id.x * 4u;
    new_values[new_pos] = sinh(f32(unpacked[0]));
    new_values[new_pos + 1u] = sinh(f32(unpacked[1]));
    new_values[new_pos + 2u] = sinh(f32(unpacked[2]));
    new_values[new_pos + 3u] = sinh(f32(unpacked[3]));
}