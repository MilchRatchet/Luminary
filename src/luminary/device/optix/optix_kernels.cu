#define OPTIX_KERNEL

#define OPTIX_PAYLOAD_DEPTH 0
#define OPTIX_PAYLOAD_TRIANGLE_HANDLE 1

#include "math.cuh"
#include "memory.cuh"
#include "optix_include.cuh"
#include "trace.cuh"
#include "utils.cuh"

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < trace_task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const int offset      = get_task_address(i);
    const DeviceTask task = task_load(offset);

    uint32_t instance_id;
    const float tmax = trace_preprocess(task, instance_id);

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    unsigned int depth = __float_as_uint(tmax);

    TriangleHandle handle = triangle_handle_get(instance_id, 0);

    OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_DEPTH, 0);
    OPTIX_PAYLOAD_INDEX_REQUIRE(OPTIX_PAYLOAD_TRIANGLE_HANDLE, 1);
    optixTrace(
      device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), 0, OPTIX_SBT_OFFSET_GEOMETRY_TRACE, 0, 0, depth,
      handle.instance_id, handle.tri_id);

    triangle_handle_store(handle, offset);
    trace_depth_store(__uint_as_float(depth), offset);
  }
}
