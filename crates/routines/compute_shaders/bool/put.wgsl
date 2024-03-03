@group(0)
@binding(0)
var<storage, read> src_values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> dst_values: array<atomic<u32>>;

@group(0)
@binding(2)
var<storage, read> src_indexes: array<u32>;

@group(0)
@binding(3)
var<storage, read> dst_indexes: array<u32>;

@compute
@workgroup_size(256)
fn put(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_src_index = src_indexes[global_id.x] / 32u;
    let base_dst_index = dst_indexes[global_id.x] / 32u;
    let src_index = src_indexes[global_id.x] % 32u;
    let dst_index = dst_indexes[global_id.x] % 32u;
    var mask = (1u << src_index);
    var value = src_values[base_src_index] & mask;
    mask = ~(1u << dst_index);
    if src_index > dst_index {
        value = value >> (src_index - dst_index);
    } else {
        value = value << (dst_index - src_index);
    }
    atomicAnd(&dst_values[base_dst_index], mask);
    atomicOr(&dst_values[base_dst_index], value);
}