#ifndef CU_LUMINARY_OPTIX_ANYHIT_H
#define CU_LUMINARY_OPTIX_ANYHIT_H

#if defined(OPTIX_KERNEL)

#include "optix_common.cuh"
#include "optix_utils.cuh"

#define OPTIX_ANYHIT_FUNC_NAME(X) __anyhit__##X

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_GEOMETRY_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(geometry_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_GEOMETRY_TRACE);

  const TriangleHandle handle        = optixGetTriangleHandle();
  const TriangleHandle ignore_handle = optixGetPayloadTriangleHandle(OPTIX_KERNEL_FUNCTION_GEOMETRY_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE);

  // Ignore self intersection if requested
  if (triangle_handle_equal(handle, ignore_handle)) {
    optixIgnoreIntersection();
  }

  const OptixAlphaResult alpha_result = optix_alpha_test(handle);

  if (alpha_result == OPTIX_ALPHA_RESULT_TRANSPARENT) {
    optixIgnoreIntersection();
  }
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_PARTICLE_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(particle_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_PARTICLE_TRACE);

  if (particle_opacity_cutout(optixGetTriangleBarycentrics())) {
    optixIgnoreIntersection();
  }
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(shadow_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_TRACE);

  const TriangleHandle handle       = optixGetTriangleHandle();
  const TriangleHandle target_light = optixGetPayloadTriangleHandle(OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE);

  if (triangle_handle_equal(handle, target_light)) {
    optixIgnoreIntersection();
  }

  const TriangleHandle ignore_handle = optixGetPayloadTriangleHandle(OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE3);

  if (triangle_handle_equal(handle, ignore_handle)) {
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

  const bool is_colored_transparency = material.flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY;

  if (albedo.a == 0.0f && !is_colored_transparency) {
    optixIgnoreIntersection();
  }

  const float transparency = 1.0f - albedo.a;

  const RGBF throughput = is_colored_transparency ? scale_color(opaque_color(albedo), transparency) : splat_color(transparency);

  RGBF total_throughput = optixGetPayloadColor(OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT);
  total_throughput      = mul_color(total_throughput, throughput);

  optixSetPayloadColor(OPTIX_KERNEL_FUNCTION_SHADOW_TRACE_PAYLOAD_VALUE_THROUGHPUT, total_throughput);

  optixIgnoreIntersection();
}

////////////////////////////////////////////////////////////////////
// OPTIX_SBT_OFFSET_SHADOW_SUN_TRACE
////////////////////////////////////////////////////////////////////

extern "C" __global__ void OPTIX_ANYHIT_FUNC_NAME(shadow_sun_trace)() {
  optixSetPayloadTypes(OPTIX_KERNEL_FUNCTION_PAYLOAD_TYPE_ID_SHADOW_SUN_TRACE);

  const TriangleHandle handle        = optixGetTriangleHandle();
  const TriangleHandle ignore_handle = optixGetPayloadTriangleHandle(OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_TRIANGLE_HANDLE);

  if (triangle_handle_equal(handle, ignore_handle)) {
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

  const bool is_colored_transparency = material.flags & DEVICE_MATERIAL_FLAG_COLORED_TRANSPARENCY;

  if (albedo.a == 0.0f && !is_colored_transparency) {
    optixIgnoreIntersection();
  }

  const float transparency = 1.0f - albedo.a;

  const RGBF throughput = is_colored_transparency ? scale_color(opaque_color(albedo), transparency) : splat_color(transparency);

  RGBF total_throughput = optixGetPayloadColor(OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT);
  total_throughput      = mul_color(total_throughput, throughput);

  optixSetPayloadColor(OPTIX_KERNEL_FUNCTION_SHADOW_SUN_TRACE_PAYLOAD_VALUE_THROUGHPUT, total_throughput);

  optixIgnoreIntersection();
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_OPTIX_ANYHIT_H */
