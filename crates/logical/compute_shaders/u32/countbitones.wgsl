@group(0)
@binding(0)
var<storage, read> input: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> output: array<u32>;

@compute
@workgroup_size(256)
fn countob(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&input) {
        output[global_id.x] = countOneBits(input[global_id.x]);
    }
}
