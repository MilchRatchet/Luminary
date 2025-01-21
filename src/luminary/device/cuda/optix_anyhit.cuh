#ifndef CU_LUMINARY_OPTIX_ANYHIT_H
#define CU_LUMINARY_OPTIX_ANYHIT_H

#include "optix_common.cuh"
#include "optix_utils.cuh"

#define OPTIX_ANYHIT_FUNC_NAME(X) __anyhit__##X

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_GEOMETRY_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(geometry_trace)() {
  // TODO: Add OMMs again.
  const TriangleHandle handle = optixGetTriangleHandle();

  const OptixAlphaResult alpha_result = optix_alpha_test(handle);

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }
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

  // Currently, materials are just two loads so it makes no sense to only load parts of it.
  const DeviceMaterial material = load_material(device.ptrs.materials, material_id);

  const RGBAF albedo = optix_get_albedo_for_shadowing(handle, material);

  if (albedo.a == 1.0f) {
    optixTerminateRay();
  }

  if (albedo.a == 0.0f) {
    optixIgnoreIntersection();
  }

  const RGBF alpha = (material.flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY) ? scale_color(opaque_color(albedo), albedo.a)
                                                                                  : get_color(albedo.a, albedo.a, albedo.a);

  CompressedAlpha compressed_alpha = optixGetPayloadCompressedAlpha();

  RGBF accumulated_alpha = optix_decompress_color(compressed_alpha);
  accumulated_alpha      = mul_color(accumulated_alpha, alpha);
  compressed_alpha       = optix_compress_color(accumulated_alpha);

  optixSetPayloadCompressedAlpha(compressed_alpha);

  optixIgnoreIntersection();
}

#endif /* CU_LUMINARY_OPTIX_ANYHIT_H */
