@group(0)
@binding(0)
var<storage, read> left_values: array<u32>;

@group(0)
@binding(1)
var<storage, write> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn u32_and_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = left_values[global_id.x] & right_values[global_id.x];
}