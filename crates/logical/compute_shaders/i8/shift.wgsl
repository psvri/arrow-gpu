@group(0)
@binding(0)
var<storage, read> left_values: array<u32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<u32>;

@compute
@workgroup_size(256)
fn bitwise_shl(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&new_values) {
        let lhs = unpack4xI8(left_values[global_id.x]);
        let index_pos = (global_id.x * 4u);
        let rhs = vec4(
            right_values[index_pos], 
            right_values[index_pos + 1], 
            right_values[index_pos + 2], 
            right_values[index_pos + 3]
        );

        new_values[global_id.x] = pack4xI8(lhs << rhs);
    }
}

@compute
@workgroup_size(256)
fn bitwise_shr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&new_values) {
        let lhs = unpack4xI8(left_values[global_id.x]);
        let index_pos = (global_id.x * 4u);
        let rhs = vec4(
            right_values[index_pos], 
            right_values[index_pos + 1], 
            right_values[index_pos + 2], 
            right_values[index_pos + 3]
        );

        new_values[global_id.x] = pack4xI8(lhs >> rhs);
    }
}