@group(0)
@binding(0)
var<storage, read> input_values: array<f32>;

@group(0)
@binding(1)
var<storage, read> power_values: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> output_values: array<f32>;


@compute
@workgroup_size(256)
fn power_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output_values[global_id.x] = pow(input_values[global_id.x], power_values[global_id.x]);
}