@group(0)
@binding(0)
var<storage, read> original_values: array<f32>;

@group(0)
@binding(1)
var<storage, write> new_values: array<f32>;

@group(0)
@binding(2)
var<storage, read> increment: f32;

@compute
@workgroup_size(1)
fn f32_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    new_values[global_id.x] = original_values[global_id.x] + increment;
}