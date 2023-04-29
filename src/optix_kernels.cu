#define UTILS_NO_DEVICE_TABLE
#define RANDOM_NO_KERNELS

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

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

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.x;

    unsigned int depth  = __float_as_uint(result.x);
    unsigned int hit_id = __float_as_uint(result.y);
    unsigned int cost   = 0;

    optixTrace(device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), ray_flags, 0, 0, 0, depth, hit_id, cost);

    float2 trace_result;

    if (device.shading_mode == SHADING_HEAT) {
      trace_result = make_float2(cost, __uint_as_float(hit_id));
    }
    else {
      trace_result = make_float2(__uint_as_float(depth), __uint_as_float(hit_id));
    }

    __stcs((float2*) (device.ptrs.trace_results + offset), trace_result);
  }
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ int perform_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t maps = device.scene.triangles[hit_id].object_maps;
  const uint16_t tex  = device.scene.texture_assignments[maps].albedo_map;

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float alpha = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v).w;

    if (alpha <= device.scene.material.alpha_cutoff) {
      return 2;
    }

    if (alpha < 1.0f) {
      return 1;
    }
  }

  return 0;
}

extern "C" __global__ void __anyhit__optix() {
  optixSetPayload_2(optixGetPayload_2() + 1);

  const int alpha_test = perform_alpha_test();

  if (alpha_test == 2) {
    optixIgnoreIntersection();
  }

  if (device.iteration_type == TYPE_LIGHT && alpha_test == 0) {
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(REJECT_HIT);

    optixTerminateRay();
  }
}

extern "C" __global__ void __closesthit__optix() {
  if (device.iteration_type == TYPE_LIGHT) {
    // If anyhit was never executed, then we must have hit fully opaque geometry.
    if (!optixGetPayload_2()) {
      optixSetPayload_0(__float_as_uint(0.0f));
      optixSetPayload_1(REJECT_HIT);
      return;
    }
  }

  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}
