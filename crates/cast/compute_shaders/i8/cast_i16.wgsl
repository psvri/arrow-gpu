@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn cast_i16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let unpacked = unpack4xI8(original_values[global_id.x]);
    let new_pos = global_id.x * 2u;
    new_values[new_pos] = ((unpacked[0] & 0x0000ffff) | (unpacked[1] << 16u));
    new_values[new_pos + 1u] = ((unpacked[2] & 0x0000ffff) | (unpacked[3] << 16u));
}