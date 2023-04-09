@group(0)
@binding(0)
var<storage, read> left_values: array<i32>;

@group(0)
@binding(1)
var<storage, write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn bitwise_not(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = ~left_values[global_id.x];
}