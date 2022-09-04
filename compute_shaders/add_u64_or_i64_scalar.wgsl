@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read> increment: vec2<u32>;

@compute
@workgroup_size(2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // The values are in little endian format
    // Hence we need to carry over the overflow bit in the first half    
    if (global_id.x % 2u == 0u) {
        var lhs_result: u32 = v_indices[global_id.x] + increment.x;
        if (lhs_result < v_indices[global_id.x] || lhs_result < increment.x) {
            v_indices[global_id.x + 1u] = v_indices[global_id.x + 1u] + 1u;
        }
        v_indices[global_id.x] = lhs_result;
    } else {
        v_indices[global_id.x] = v_indices[global_id.x] + increment.y;
    }
}


