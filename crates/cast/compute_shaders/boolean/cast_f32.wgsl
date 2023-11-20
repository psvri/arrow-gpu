@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<f32>;

@compute
@workgroup_size(256)
fn cast_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {

    if global_id.x < arrayLength(&new_values) {
        let index_by_32 = global_id.x / 32u;
        let bit_pos = (1u << (global_id.x % 32u));
        if (original_values[index_by_32] & bit_pos) == bit_pos {
            new_values[global_id.x] = 1.0;
        }
    }
}