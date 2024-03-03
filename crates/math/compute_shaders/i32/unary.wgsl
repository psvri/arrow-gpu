@group(0)
@binding(0)
var<storage, read> original_values: array<i32>;

@group(0)
@binding(1)
var<storage, read_write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn abs_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = abs(original_values[global_id.x]);
}