@group(0)
@binding(0)
var<storage, read> src_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> dst_values: array<u32>;

@group(0)
@binding(2)
var<storage, read> src_indexes: array<u32>;

@group(0)
@binding(3)
var<storage, read> dst_indexes: array<u32>;

@compute
@workgroup_size(256)
fn put(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < arrayLength(&src_indexes)) {
        dst_values[dst_indexes[global_id.x]] = src_values[src_indexes[global_id.x]];
    }
}