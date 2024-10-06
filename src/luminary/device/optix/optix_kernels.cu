#define OPTIX_KERNEL

#include "bvh_utils.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "trace.cuh"
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
    const int offset         = get_task_address(i);
    const TraceTask task     = load_trace_task(device.ptrs.trace_tasks + offset);
    const TraceResult result = trace_preprocess(task);

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.depth;

    unsigned int depth  = __float_as_uint(result.depth);
    unsigned int hit_id = result.hit_id;

    optixTrace(device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), 0, 0, 0, 0, depth, hit_id);

    float2 trace_result;

    if (device.settings.shading_mode == LUMINARY_SHADING_MODE_HEAT) {
      trace_result = make_float2(0.0f, __uint_as_float(hit_id));
    }
    else {
      trace_result = make_float2(__uint_as_float(depth), __uint_as_float(hit_id));
    }

    __stcs((float2*) (device.ptrs.trace_results + offset), trace_result);
  }
}

// TODO: This is fucked, I need to fix it by checking for IOR aswell.
#if 0
/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ OptixAlphaResult optix_alpha_test() {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);
  const uint16_t tex         = __ldg(&(device.ptrs.materials[material_id].albedo_tex));

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float alpha = tex2D<float4>(device.ptrs.albedo_atlas[tex].handle, uv.u, 1.0f - uv.v).w;

    if (alpha == 0.0f) {
      return OPTIX_ALPHA_RESULT_TRANSPARENT;
    }

    if (alpha < 1.0f) {
      return OPTIX_ALPHA_RESULT_SEMI;
    }
  }

  return OPTIX_ALPHA_RESULT_OPAQUE;
}

extern "C" __global__ void __anyhit__optix() {

  const OptixAlphaResult alpha_result = optix_alpha_test();

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }
}
#else
extern "C" __global__ void __anyhit__optix() {
  return;
}
#endif

extern "C" __global__ void __closesthit__optix() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}
