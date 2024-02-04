@group(0)
@binding(0)
var<storage, read> input_data: array<i32>;

@group(0)
@binding(1)
var<storage, read_write> output_data: array<i32>;

const wg_size = 256u;

var<workgroup> shared_data: array<i32, wg_size>;

@compute
@workgroup_size(256)
fn sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if global_id.x >= arrayLength(&input_data) {
        return;
    }

    shared_data[local_id.x] = input_data[global_id.x];
    workgroupBarrier();

    for (var s = 1u; s < wg_size; s *= 2u) {
        if (local_id.x % (2u*s) == 0) && (global_id.x + s < arrayLength(&input_data)) {
            shared_data[local_id.x] += shared_data[local_id.x + s];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        output_data[wg_id.x] = shared_data[0];
    }
}