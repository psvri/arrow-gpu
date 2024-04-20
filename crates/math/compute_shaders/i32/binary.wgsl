@group(0)
@binding(0)
var<storage, read> input_values: array<i32>;

@group(0)
@binding(1)
var<storage, read_write> power_values: array<i32>;

@group(0)
@binding(2)
var<storage, read_write> output_values: array<i32>;

@compute
@workgroup_size(256)
fn power_(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var input_value = input_values[global_id.x];
    var result = 1;
    if power_values[global_id.x] >= 0 {
        for (var i = 0; i < power_values[global_id.x]; i++) {
            result *= input_value;
        }
    } else {
        for (var i = 0; i < abs(power_values[global_id.x]); i++) {
            result /= input_value;
        }
    }

    output_values[global_id.x] = result;
}