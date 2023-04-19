@group(0)
@binding(0)
var<storage, read> left_values: array<i32>;

@group(0)
@binding(1)
var<storage, read> right_values: array<i32>;

@group(0)
@binding(2)
var<storage, read_write> new_values: array<u32>;

// size = 256/8
var<workgroup> local_set_bits: array<atomic<u32>, 32>;

fn set_bit(index: u32, value: bool) {
    let index_by_32 = index / 32u;
    if value {
        atomicOr(&local_set_bits[index_by_32], (1u << (index % 32u)));
    }
}

@compute
@workgroup_size(256)
fn gt(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index_pos = local_id.x * 4u;
    var value = get_left_byte(left_values[global_id.x]) > get_left_byte(right_values[global_id.x]);
    set_bit(index_pos, value);
    value = get_mid_left_byte(left_values[global_id.x]) > get_mid_left_byte(right_values[global_id.x]);
    set_bit(index_pos + 1u, value);
    value = get_mid_right_byte(left_values[global_id.x]) > get_mid_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 2u, value);
    value = get_right_byte(left_values[global_id.x]) > get_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 3u, value);
    workgroupBarrier();
    if global_id.x % 8u == 0u {
        new_values[global_id.x / 8u] = atomicLoad(&local_set_bits[local_id.x / 8u]);
    }
}

@compute
@workgroup_size(256)
fn gteq(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index_pos = local_id.x * 4u;
    var value = get_left_byte(left_values[global_id.x]) >= get_left_byte(right_values[global_id.x]);
    set_bit(index_pos, value);
    value = get_mid_left_byte(left_values[global_id.x]) >= get_mid_left_byte(right_values[global_id.x]);
    set_bit(index_pos + 1u, value);
    value = get_mid_right_byte(left_values[global_id.x]) >= get_mid_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 2u, value);
    value = get_right_byte(left_values[global_id.x]) >= get_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 3u, value);
    workgroupBarrier();
    if global_id.x % 8u == 0u {
        new_values[global_id.x / 8u] = atomicLoad(&local_set_bits[local_id.x / 8u]);
    }
}

@compute
@workgroup_size(256)
fn lt(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index_pos = local_id.x * 4u;
    var value = get_left_byte(left_values[global_id.x]) < get_left_byte(right_values[global_id.x]);
    set_bit(index_pos, value);
    value = get_mid_left_byte(left_values[global_id.x]) < get_mid_left_byte(right_values[global_id.x]);
    set_bit(index_pos + 1u, value);
    value = get_mid_right_byte(left_values[global_id.x]) < get_mid_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 2u, value);
    value = get_right_byte(left_values[global_id.x]) < get_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 3u, value);
    workgroupBarrier();
    if global_id.x % 8u == 0u {
        new_values[global_id.x / 8u] = atomicLoad(&local_set_bits[local_id.x / 8u]);
    }
}

@compute
@workgroup_size(256)
fn lteq(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index_pos = local_id.x * 4u;
    var value = get_left_byte(left_values[global_id.x]) <= get_left_byte(right_values[global_id.x]);
    set_bit(index_pos, value);
    value = get_mid_left_byte(left_values[global_id.x]) <= get_mid_left_byte(right_values[global_id.x]);
    set_bit(index_pos + 1u, value);
    value = get_mid_right_byte(left_values[global_id.x]) <= get_mid_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 2u, value);
    value = get_right_byte(left_values[global_id.x]) <= get_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 3u, value);
    workgroupBarrier();
    if global_id.x % 8u == 0u {
        new_values[global_id.x / 8u] = atomicLoad(&local_set_bits[local_id.x / 8u]);
    }
}

@compute
@workgroup_size(256)
fn eq(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index_pos = local_id.x * 4u;
    var value = get_left_byte(left_values[global_id.x]) == get_left_byte(right_values[global_id.x]);
    set_bit(index_pos, value);
    value = get_mid_left_byte(left_values[global_id.x]) == get_mid_left_byte(right_values[global_id.x]);
    set_bit(index_pos + 1u, value);
    value = get_mid_right_byte(left_values[global_id.x]) == get_mid_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 2u, value);
    value = get_right_byte(left_values[global_id.x]) == get_right_byte(right_values[global_id.x]);
    set_bit(index_pos + 3u, value);
    workgroupBarrier();
    if global_id.x % 8u == 0u {
        new_values[global_id.x / 8u] = atomicLoad(&local_set_bits[local_id.x / 8u]);
    }
}