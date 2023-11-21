@group(0)
@binding(0)
var<storage, read> values: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> result: atomic<u32>;

var<workgroup> workgroup_result: atomic<u32>;

@compute
@workgroup_size(256)
fn any(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if values[global_id.x] > 0u {
        atomicAdd(&workgroup_result, 1u);
    }
    workgroupBarrier();
    if local_id.x == 0u && atomicLoad(&workgroup_result) > 0u {
        atomicAdd(&result, 1u);
    }
}