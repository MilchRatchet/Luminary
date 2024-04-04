#define UTILS_NO_DEVICE_TABLE

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

#define MAX_COMPRESSABLE_COLOR (1.99999988079071044921875f)

__device__ void optix_compress_color(RGBF color, unsigned int& data0, unsigned int& data1) {
  uint32_t bits_r = (__float_as_uint(fminf(color.r + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_g = (__float_as_uint(fminf(color.g + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;
  uint32_t bits_b = (__float_as_uint(fminf(color.b + 1.0f, MAX_COMPRESSABLE_COLOR)) >> 2) & 0x1FFFFF;

  data0 = bits_r | (bits_g << 21);
  data1 = (bits_g >> 11) | (bits_b << 10);
}

__device__ RGBF optix_decompress_color(unsigned int data0, unsigned int data1) {
  uint32_t bits_r = data0 & 0x1FFFFF;
  uint32_t bits_g = (data0 >> 21) & 0x7FF | ((data1 & 0x3FF) << 11);
  uint32_t bits_b = (data1 >> 10) & 0x1FFFFF;

  RGBF color;
  color.r = __uint_as_float(0x3F800000u | (bits_r << 2)) - 1.0f;
  color.g = __uint_as_float(0x3F800000u | (bits_g << 2)) - 1.0f;
  color.b = __uint_as_float(0x3F800000u | (bits_b << 2)) - 1.0f;

  return color;
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
      // Disable OMM opaque hits because we want to know if we hit something that is fully opaque so we can reject.
      ray_flags = OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
      break;
  }

  for (int i = 0; i < trace_task_count; i++) {
    const int offset     = get_task_address2(idx.x, idx.y, i);
    const TraceTask task = load_trace_task(device.trace_tasks + offset);
    const float2 result  = __ldcs((float2*) (device.ptrs.trace_results + offset));
    const int pixel      = task.index.y * device.width + task.index.x;

    const float3 origin = make_float3(task.origin.x, task.origin.y, task.origin.z);
    const float3 ray    = make_float3(task.ray.x, task.ray.y, task.ray.z);

    const float tmax = result.x;

    if (device.iteration_type == TYPE_LIGHT) {
      unsigned int depth  = __float_as_uint(result.x);
      unsigned int hit_id = __float_as_uint(result.y);

      // 21 bits for each color component.
      unsigned int alpha_data0, alpha_data1;
      optix_compress_color(get_color(1.0f, 1.0f, 1.0f), alpha_data0, alpha_data1);

      optixTrace(
        device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), ray_flags, 0, 0, 0, depth, hit_id, alpha_data0,
        alpha_data1);

      RGBF record = load_RGBF(device.ptrs.light_records + pixel);
      record      = mul_color(record, optix_decompress_color(alpha_data0, alpha_data1));
      store_RGBF(device.ptrs.light_records + pixel, record);

      float2 trace_result = make_float2(__uint_as_float(depth), __uint_as_float(hit_id));
      __stcs((float2*) (device.ptrs.trace_results + offset), trace_result);
    }
    else {
      unsigned int depth  = __float_as_uint(result.x);
      unsigned int hit_id = __float_as_uint(result.y);
      unsigned int cost   = 0;
      unsigned int unused = 0;

      optixTrace(
        device.optix_bvh, origin, ray, 0.0f, tmax, 0.0f, OptixVisibilityMask(0xFFFF), ray_flags, 0, 0, 0, depth, hit_id, cost, unused);

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
}

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ OptixAlphaResult optix_alpha_test(RGBAF& albedo) {
  const unsigned int hit_id = optixGetPrimitiveIndex();

  const uint32_t material_id = load_triangle_material_id(hit_id);
  const uint16_t tex         = __ldg(&(device.scene.materials[material_id].albedo_map));

  albedo = get_RGBAF(0.0f, 0.0f, 0.0f, 1.0f);

  if (tex != TEXTURE_NONE) {
    const UV uv = load_triangle_tex_coords(hit_id, optixGetTriangleBarycentrics());

    const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[tex].tex, uv.u, 1.0f - uv.v);

    albedo.r = tex_value.x;
    albedo.g = tex_value.y;
    albedo.b = tex_value.z;
    albedo.a = tex_value.w;

    if (albedo.a < device.scene.material.alpha_cutoff) {
      return OPTIX_ALPHA_RESULT_TRANSPARENT;
    }

    if (albedo.a < 1.0f) {
      return OPTIX_ALPHA_RESULT_SEMI;
    }
  }

  return OPTIX_ALPHA_RESULT_OPAQUE;
}

extern "C" __global__ void __anyhit__optix() {
  if (device.iteration_type == TYPE_CAMERA) {
    optixSetPayload_2(optixGetPayload_2() + 1);
  }

  RGBAF albedo;
  const OptixAlphaResult alpha_result = optix_alpha_test(albedo);

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }

  if (device.iteration_type == TYPE_LIGHT) {
    if (optixGetPrimitiveIndex() == optixGetPayload_1()) {
      optixIgnoreIntersection();
    }

    if (alpha_result == OPTIX_ALPHA_RESULT_OPAQUE) {
      optixSetPayload_0(__float_as_uint(0.0f));
      optixSetPayload_1(HIT_TYPE_REJECT);

      optixTerminateRay();
    }

    RGBF alpha = (device.scene.material.colored_transparency) ? scale_color(opaque_color(albedo), 1.0f - albedo.a)
                                                              : get_color(1.0f - albedo.a, 1.0f - albedo.a, 1.0f - albedo.a);

    unsigned int alpha_data0 = optixGetPayload_2();
    unsigned int alpha_data1 = optixGetPayload_3();

    RGBF accumulated_alpha = optix_decompress_color(alpha_data0, alpha_data1);
    accumulated_alpha      = mul_color(accumulated_alpha, alpha);
    optix_compress_color(accumulated_alpha, alpha_data0, alpha_data1);

    optixSetPayload_2(alpha_data0);
    optixSetPayload_3(alpha_data1);

    optixIgnoreIntersection();
  }
}

extern "C" __global__ void __closesthit__optix() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
}
