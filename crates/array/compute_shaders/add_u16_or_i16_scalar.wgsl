@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read> increment: u32;

let MAX_u16: u32 = 0xffffu;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let result_1: u32 = (v_indices[global_id.x] + increment) & MAX_u16;
    let result_2: u32 = (v_indices[global_id.x] + (increment << 16u)) & (MAX_u16 << 16u);
    v_indices[global_id.x] = result_1 + result_2;
}