@group(0)
@binding(0)
var<storage, read> left_values: array<f32>;

@group(0)
@binding(1)
var<storage, read> array_size: u32;


@group(0)
@binding(2)
var<storage, read_write> new_values: array<f32>;

const wg_size = 256u;

var<workgroup> local_sum: array<f32, wg_size>;

@compute
@workgroup_size(256)
fn sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if global_id.x >= array_size {
        return;
    }

    local_sum[local_id.x] = left_values[global_id.x];
    workgroupBarrier();

    for (var s = 1u; s < wg_size; s *= 2u) {
        if (local_id.x % (2u * s) == 0u) && (global_id.x + s < array_size) {
            local_sum[local_id.x] += local_sum[local_id.x + s];
        }
        workgroupBarrier();
    }

    if local_id.x == 0u {
        new_values[wg_id.x] = local_sum[local_id.x];
    }
}