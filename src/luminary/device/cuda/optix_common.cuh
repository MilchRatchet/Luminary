#ifndef CU_LUMINARY_OPTIX_COMMON_H
#define CU_LUMINARY_OPTIX_COMMON_H

#if defined(OPTIX_KERNEL)

#include "ior_stack.cuh"
#include "memory.cuh"
#include "optix_utils.cuh"
#include "utils.cuh"

enum OptixAlphaResult {
  OPTIX_ALPHA_RESULT_OPAQUE      = 0,
  OPTIX_ALPHA_RESULT_SEMI        = 1,
  OPTIX_ALPHA_RESULT_TRANSPARENT = 2
} typedef OptixAlphaResult;

/*
 * Performs alpha test on triangle
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
LUMINARY_FUNCTION OptixAlphaResult optix_alpha_test(const TriangleHandle handle) {
  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const uint16_t material_id = material_id_load(mesh_id, handle.tri_id);
  const uint16_t tex         = __ldg(&(device.ptrs.materials[material_id].albedo_tex));

  if (tex != TEXTURE_NONE) {
    const DeviceTextureObject texture = load_texture_object(tex);

    if (texture_is_valid(texture) == false)
      return OPTIX_ALPHA_RESULT_OPAQUE;

    const UV uv = load_triangle_tex_coords(handle, optixGetTriangleBarycentrics());

    const float alpha = texture_load(texture, uv).w;

    if (alpha == 0.0f) {
      return OPTIX_ALPHA_RESULT_TRANSPARENT;
    }

    if (alpha < 1.0f) {
      return OPTIX_ALPHA_RESULT_SEMI;
    }
  }

  return OPTIX_ALPHA_RESULT_OPAQUE;
}

LUMINARY_FUNCTION RGBAF optix_get_albedo_for_shadowing(const TriangleHandle handle, const DeviceMaterial material) {
  RGBAF albedo = material.albedo;

  if (material.albedo_tex != TEXTURE_NONE) {
    const DeviceTextureObject texture = load_texture_object(material.albedo_tex);

    if (texture_is_valid(texture) == false)
      return get_RGBAF(0.9f, 0.9f, 0.9f, 1.0f);

    const UV uv = load_triangle_tex_coords(handle, optixGetTriangleBarycentrics());

    const float4 tex_value = texture_load(texture, uv);

    albedo = get_RGBAF(tex_value.x, tex_value.y, tex_value.z, tex_value.w);
  }

  return albedo;
}

LUMINARY_FUNCTION bool particle_opacity_cutout(const float2 coord) {
  const float dx = coord.x - 0.5f;
  const float dy = coord.y - 0.5f;

  const float r = dx * dx + dy * dy;

  return (r > 0.25f);
}

LUMINARY_FUNCTION RGBF optix_geometry_shadowing(
  const TriangleHandle handle, const vec3 position, const vec3 dir, const float dist, TriangleHandle target_light,
  const OptixTraceStatus status) {
  OptixKernelFunctionShadowTracePayload payload;
  payload.handle        = target_light;
  payload.throughput    = splat_color(1.0f);
  payload.handle_origin = handle;

  optixKernelFunctionShadowTrace(
    device.optix_bvh_shadow, position, dir, eps, dist, 0.0f, OptixVisibilityMask(0xFFFF),
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, status, payload);

  if (payload.handle.instance_id == HIT_TYPE_REJECT) {
    payload.throughput = get_color(0.0f, 0.0f, 0.0f);
  }

  return payload.throughput;
}

LUMINARY_FUNCTION RGBF
  optix_sun_shadowing(const TriangleHandle handle, const vec3 position, const vec3 dir, const float dist, const OptixTraceStatus status) {
  OptixKernelFunctionShadowSunTracePayload payload;
  payload.throughput    = splat_color(1.0f);
  payload.handle_origin = handle;

  optixKernelFunctionShadowSunTrace(
    device.optix_bvh_shadow, position, dir, eps, dist, 0.0f, OptixVisibilityMask(0xFFFF),
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, status, payload);

  return payload.throughput;
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_OPTIX_COMMON_H */
