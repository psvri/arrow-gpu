@group(0)
@binding(0)
var<storage, read> input_data: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> output_data: array<u32>;

const wg_size = 128u;

var<workgroup> shared_data: array<u32, wg_size>;

@compute
@workgroup_size(128)
fn sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if global_id.x >= arrayLength(&input_data) {
        return;
    }

    var tid = local_id.x;
    var i = wg_id.x * (wg_size * 2) + tid;
    if (i + wg_size) < arrayLength(&input_data) {
        shared_data[tid] = input_data[i] + input_data[i + wg_size];
    } else {
        shared_data[tid] = input_data[i] ;
    }
    workgroupBarrier();

    for (var s = wg_size / 2u; s > 0u; s >>= 1u) {
        if (local_id.x < s) && (i < arrayLength(&input_data)) {
            shared_data[local_id.x] += shared_data[local_id.x + s];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        output_data[wg_id.x] = shared_data[0];
    }
}