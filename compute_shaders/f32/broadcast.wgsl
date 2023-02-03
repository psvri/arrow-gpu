@group(0)
@binding(0)
var<storage, write> new_values: array<f32>;

@group(0)
@binding(1)
var<storage, read> operand: f32;

@compute
@workgroup_size(256)
fn broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = operand;
}