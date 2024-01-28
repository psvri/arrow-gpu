@group(0)
@binding(0)
var<storage, read> left_values: array<i32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<i32>;

@compute
@workgroup_size(256)
fn bitwise_shl(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_pos = (global_id.x * 2u);
    var left_u16 = get_left_half(left_values[global_id.x]) << right_values[index_pos];
    var right_u16 = get_right_half(left_values[global_id.x]) << right_values[index_pos + 1u];
    new_values[global_id.x] = (left_u16 & MAX_I16) | ((right_u16 & MAX_I16) << 16u);
}

@compute
@workgroup_size(256)
fn bitwise_shr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_pos = (global_id.x * 2u);
    var left_u16 = shr(get_left_half(left_values[global_id.x]), right_values[index_pos]);
    var right_u16 = shr(get_right_half(left_values[global_id.x]), right_values[index_pos + 1u]);
    new_values[global_id.x] = (left_u16 & MAX_I16) | ((right_u16 & MAX_I16) << 16u) ;
}

// i32(0x00008000) is done temporarily until https://github.com/gfx-rs/wgpu/issues/5085 is resolved
fn shr(input: i32, shift_value: u32) -> i32 {
    if input < 0 {
        var result = input >> shift_value;
        var other_bit_sets = MAX_I16 << (16u - shift_value);
        other_bit_sets = other_bit_sets | (i32(0x00008000) >> shift_value);
        return other_bit_sets | result;
    } else {
        return input >> shift_value;
    }
}