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
    let start_index = global_id.x * 32;
    var result = 0u;
    for (var i = 0u; i < 32 & (start_index + i) < arrayLength(&indexes); i++) {
        let index = indexes[start_index + i];
        let base_src_index = index / 32u;
        let src_index = index % 32u;
        var value = left_values[base_src_index] & (1u << src_index);
        if src_index > i {
            value = value >> (src_index - i);
        } else {
            value = value << (i - src_index);
        }
        result = result | value;
    }
    new_values[global_id.x] = result;
}