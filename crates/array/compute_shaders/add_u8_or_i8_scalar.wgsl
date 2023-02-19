@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read> increment: u32;

let MAX_u8: u32 = 0xffu;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let result_1: u32 = (v_indices[global_id.x] + increment) & MAX_u8;
    let result_2: u32 = (v_indices[global_id.x] + (increment << 8u)) & (MAX_u8 << 8u);
    let result_3: u32 = (v_indices[global_id.x] + (increment << 16u)) & (MAX_u8 << 16u);
    let result_4: u32 = (v_indices[global_id.x] + (increment << 24u)) & (MAX_u8 << 24u);
    v_indices[global_id.x] = result_1 + result_2 + result_3 + result_4;
}