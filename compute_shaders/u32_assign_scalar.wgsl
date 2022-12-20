@group(0)
@binding(0)
var<storage, read_write> values: array<u32>;

@group(0)
@binding(1)
var<storage, read> increment: u32;

@compute
@workgroup_size(256)
fn u32_add_assign(@builtin(global_invocation_id) global_id: vec3<u32>) {
    values[global_id.x] = values[global_id.x] + increment;
}