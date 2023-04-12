#define UTILS_NO_DEVICE_TABLE
#define RANDOM_NO_KERNELS

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "math.cuh"
#include "utils.cuh"

__device__ TraceTask load_trace_task_essentials_optix(const void* ptr) {
  const float4 data0 = __ldcs((float4*) ptr);
  const float2 data1 = __ldcs(((float2*) ptr) + 2);

  TraceTask task;
  task.origin.x = data0.x;
  task.origin.y = data0.y;
  task.origin.z = data0.z;
  task.ray.x    = data0.w;

  task.ray.y = data1.x;
  task.ray.z = data1.y;

  return task;
}

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  const uint3 idx  = optixGetLaunchIndex();
  const uint3 dimx = optixGetLaunchDimensions();

  const uint16_t trace_task_count = device.trace_count[idx.x + idx.y * dimx.x];

  unsigned int ray_flags;

  switch (device.iteration_type) {
    case TYPE_CAMERA:
      ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
      break;
    case TYPE_LIGHT:
      ray_flags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
      break;
    default:
    case TYPE_BOUNCE:
      ray_flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
      break;
  }

  for (int i = 0; i < trace_task_count; i++) {
    const int offset     = get_task_address2(idx.x, idx.y, i);
    const TraceTask task = load_trace_task_essentials_optix(device.trace_tasks + offset);
    const float2 result  = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.x - eps * fabsf(result.x);

    unsigned int depth  = __float_as_uint(result.x);
    unsigned int hit_id = __float_as_uint(result.y);

    optixTrace(device.optix_bvh, origin, ray, eps, tmax, 0.0f, OptixVisibilityMask(0xFFFF), ray_flags, 0, 0, 0, depth, hit_id);

    const float2 trace_result = make_float2(__uint_as_float(depth), __uint_as_float(hit_id));
    __stcs((float2*) (device.ptrs.trace_results + offset), trace_result);
  }
}

extern "C" __global__ void __anyhit__optix() {
  optixSetPayload_0(__float_as_uint(0.0f));
  optixSetPayload_1(REJECT_HIT);

  optixTerminateRay();
}

extern "C" __global__ void __closesthit__optix() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__optix() {
}
