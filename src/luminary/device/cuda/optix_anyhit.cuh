#ifndef CU_LUMINARY_OPTIX_ANYHIT_H
#define CU_LUMINARY_OPTIX_ANYHIT_H

#include "optix_common.cuh"
#include "optix_utils.cuh"

#define OPTIX_ANYHIT_FUNC_NAME(X) __anyhit__##X

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_GEOMETRY_TRACE
////////////////////////////////////////////////////////////////////

// TODO: This is fucked, I need to fix it by checking for IOR aswell.

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(geometry_trace)() {
#if 0
  const OptixAlphaResult alpha_result = optix_alpha_test();

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }
#endif
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_PARTICLE_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(particle_trace)() {
  if (particle_opacity_cutout(optixGetTriangleBarycentrics())) {
    optixIgnoreIntersection();
  }
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_LIGHT_BSDF_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(light_bsdf_trace)() {
  // TODO: Add support to check for fully transparent hits, we want to ignore those.
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(shadow_trace)() {
  const TriangleHandle handle = optixGetTriangleHandle();

  const TriangleHandle target_light = optixGetPayloadTriangleHandle();

  if (triangle_handle_equal(handle, target_light)) {
    optixIgnoreIntersection();
  }

  const uint32_t mesh_id = mesh_id_load(handle.instance_id);

  const uint16_t material_id = material_id_load(mesh_id, handle.tri_id);

  const bool material_has_ior_shadowing = (device.ptrs.materials[material_id].flags & DEVICE_MATERIAL_FLAG_IOR_SHADOWING) != 0;

  unsigned int ray_ior = (!material_has_ior_shadowing) ? SKIP_IOR_CHECK : optixGetPayloadIOR();

  const RGBAF albedo = optix_get_albedo_with_ior_check(handle, material_id, ray_ior);

  if (albedo.a == 1.0f) {
    optixSetPayloadGeneric(OPTIX_PAYLOAD_TRIANGLE_HANDLE, HIT_TYPE_REJECT);

    optixTerminateRay();
  }

  int8_t ray_ior_pop_balance = (int8_t) ((ray_ior >> 16) & 0xFF);
  int8_t ray_ior_pop_max     = (int8_t) (ray_ior >> 24);

  ray_ior_pop_balance += (optixIsBackFaceHit()) ? 1 : -1;
  ray_ior_pop_max = max(ray_ior_pop_max, ray_ior_pop_balance);

  ray_ior = (ray_ior & 0xFFFF) | (uint32_t) (ray_ior_pop_balance << 16) | (uint32_t) (ray_ior_pop_max << 24);

  optixSetPayloadGeneric(OPTIX_PAYLOAD_IOR, ray_ior);

  if (albedo.a == 0.0f) {
    optixIgnoreIntersection();
  }

  const RGBF alpha = (device.ptrs.materials[material_id].flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY)
                       ? scale_color(opaque_color(albedo), 1.0f - albedo.a)
                       : get_color(1.0f - albedo.a, 1.0f - albedo.a, 1.0f - albedo.a);

  CompressedAlpha compressed_alpha = optixGetPayloadCompressedAlpha();

  RGBF accumulated_alpha = optix_decompress_color(compressed_alpha);
  accumulated_alpha      = mul_color(accumulated_alpha, alpha);
  compressed_alpha       = optix_compress_color(accumulated_alpha);

  optixSetPayloadCompressedAlpha(compressed_alpha);

  optixIgnoreIntersection();
}

#endif /* CU_LUMINARY_OPTIX_ANYHIT_H */
