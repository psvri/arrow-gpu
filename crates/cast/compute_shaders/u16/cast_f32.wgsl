// functions get_left_half and get_right_half are present in compute_shaders/u16/utils.wgsl
// the rust code concacts them at compile time. This workaround is needed due to lack of import
// support

@group(0)
@binding(0)
var<storage, read> original_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<f32>;

@compute
@workgroup_size(256)
fn cast_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left = get_left_half(original_values[global_id.x]);
    let right = get_right_half(original_values[global_id.x]);
    let new_pos = global_id.x * 2u;
    new_values[new_pos] = f32(left);
    new_values[new_pos + 1u] = f32(right);
}