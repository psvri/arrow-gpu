@group(0)
@binding(0)
var<storage, read> left_values: array<u32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn bitwise_shl(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_pos = (global_id.x * 2u);
    var left_u16 = get_left_half(left_values[global_id.x]) << right_values[index_pos];
    var right_u16 = get_right_half(left_values[global_id.x]) << right_values[index_pos + 1u];
    new_values[global_id.x] = (left_u16 & LEFT_EXTRACTOR) | ((right_u16 << 16u) & RIGHT_EXTRACTOR);
}

@compute
@workgroup_size(256)
fn bitwise_shr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_pos = (global_id.x * 2u);
    var left_u16 = get_left_half(left_values[global_id.x]) >> right_values[index_pos];
    var right_u16 = get_right_half(left_values[global_id.x]) >> right_values[index_pos + 1u];
    new_values[global_id.x] = (left_u16 & LEFT_EXTRACTOR) | ((right_u16 << 16u) & RIGHT_EXTRACTOR);
}