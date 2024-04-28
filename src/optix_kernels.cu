#define UTILS_NO_DEVICE_TABLE

#define OPTIX_KERNEL

#include "utils.h"

extern "C" static __constant__ DeviceConstantMemory device;

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

enum OptixAlphaResult {
  OPTIX_ALPHA_RESULT_OPAQUE      = 0,
  OPTIX_ALPHA_RESULT_SEMI        = 1,
  OPTIX_ALPHA_RESULT_TRANSPARENT = 2
} typedef OptixAlphaResult;

// Kernels must be named __[SEMANTIC]__..., for example, __raygen__...
// This can be found under function name prefix in the programming guide

extern "C" __global__ void __raygen__optix() {
  const uint16_t trace_task_count = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < trace_task_count; i++) {
    const int offset     = get_task_address(i);
    const TraceTask task = load_trace_task(device.ptrs.trace_tasks + offset);
    const float2 result  = __ldcs((float2*) (device.ptrs.trace_results + offset));

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.x;

    unsigned int depth  = __float_as_uint(result.x);
    unsigned int hit_id = __float_as_uint(result.y);
    unsigned int cost   = 0;

    optixTrace(
      device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 0, 0, depth, hit_id,
      cost);

    if (__uint_as_float(depth) < tmax) {
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
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ OptixAlphaResult optix_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);
  const uint16_t tex         = __ldg(&(device.scene.materials[material_id].albedo_map));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v);

    if (tex_value.w <= device.scene.material.alpha_cutoff) {
      return OPTIX_ALPHA_RESULT_TRANSPARENT;
    }

    if (tex_value.w < 1.0f) {
      return OPTIX_ALPHA_RESULT_SEMI;
    }
  }

  return OPTIX_ALPHA_RESULT_OPAQUE;
}

extern "C" __global__ void __anyhit__optix() {
  if (IS_PRIMARY_RAY) {
    optixSetPayload_2(optixGetPayload_2() + 1);
  }

  const OptixAlphaResult alpha_result = optix_alpha_test();

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }
}

extern "C" __global__ void __closesthit__optix() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}
