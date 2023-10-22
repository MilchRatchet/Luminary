#define UTILS_NO_DEVICE_TABLE
#define RANDOM_NO_KERNELS

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

__device__ bool particle_opacity_cutout(const float2 coord) {
  const float dx = fabsf(coord.x - 0.5f);
  const float dy = fabsf(coord.y - 0.5f);

  const float r = dx * dx + dy * dy;

  return (r > 0.25f);
}

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  const uint3 idx  = optixGetLaunchIndex();
  const uint3 dimx = optixGetLaunchDimensions();

  const uint16_t trace_task_count = device.trace_count[idx.x + idx.y * dimx.x];

  unsigned int ray_flags;

  switch (device.iteration_type) {
    default:
    case TYPE_BOUNCE:
    case TYPE_CAMERA:
      ray_flags = OPTIX_RAY_FLAG_NONE;
      break;
    case TYPE_LIGHT:
      ray_flags = OPTIX_RAY_FLAG_NONE;
      break;
  }

  for (int i = 0; i < trace_task_count; i++) {
    const int offset     = get_task_address2(idx.x, idx.y, i);
    const TraceTask task = load_trace_task_essentials(device.trace_tasks + offset);
    const float2 result  = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const vec3 reference = sub_vector(task.origin, device.scene.camera.pos);

    const float3 origin = make_float3(reference.x, reference.y, reference.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.x;

    unsigned int depth  = __float_as_uint(result.x);
    unsigned int hit_id = __float_as_uint(result.y);
    unsigned int cost   = 0;

    optixTrace(
      device.optix_bvh_particles, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), ray_flags, 0, 0, 0, depth, hit_id, cost);

    if (__uint_as_float(depth) < tmax) {
      float2 trace_result = result;

      // Hit ID contains the triangle ID but we only store the actual particle / quad ID
      hit_id = HIT_TYPE_PARTICLE_MIN + (hit_id >> 1);

      if (device.shading_mode == SHADING_HEAT) {
        trace_result = make_float2(cost, __uint_as_float(hit_id));
      }
      else {
        trace_result = make_float2(__uint_as_float(depth), __uint_as_float(hit_id));
      }

      __stcs((float2*) (device.ptrs.trace_results + offset), trace_result);
    }
  }
}

extern "C" __global__ void __anyhit__optix() {
  optixSetPayload_2(optixGetPayload_2() + 1);

  if (particle_opacity_cutout(optixGetTriangleBarycentrics())) {
    optixIgnoreIntersection();
  }
}

extern "C" __global__ void __closesthit__optix() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}
