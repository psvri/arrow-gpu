@group(0)
@binding(0)
var<storage, read> left_values: array<u32>;

@group(0)
@binding(1)
var<storage, read> indexes: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn take(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = left_values[indexes[global_id.x]];
}