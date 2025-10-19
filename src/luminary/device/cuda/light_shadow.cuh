#ifndef CU_LUMINARY_LIGHT_SHADOW_H
#define CU_LUMINARY_LIGHT_SHADOW_H

#ifdef OPTIX_KERNEL

#include "optix_include.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

// #define DIRECT_LIGHTING_NO_SHADOW

struct ShadowTraceTask {
  OptixTraceStatus trace_status;
  vec3 origin;
  vec3 ray;
  float limit;
  TriangleHandle target_light;
  VolumeType volume_type;
} typedef DirectLightingShadowTask;

__device__ RGBF shadow_evaluate(const ShadowTraceTask& task, const TriangleHandle self_handle) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_geometry_shadowing(self_handle, task.origin, task.ray, task.limit, task.target_light, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.volume_type, task.origin, task.ray, task.limit));

  return visibility;
#else  /* !DIRECT_LIGHTING_NO_SHADOW */
  return splat_color(1.0f);
#endif /* DIRECT_LIGHTING_NO_SHADOW */
}

__device__ RGBF shadow_evaluate_sun(const ShadowTraceTask& task, const TriangleHandle self_handle) {
#ifndef DIRECT_LIGHTING_NO_SHADOW
  RGBF visibility = optix_sun_shadowing(self_handle, task.origin, task.ray, task.limit, task.trace_status);

  if (task.trace_status == OPTIX_TRACE_STATUS_ABORT)
    return splat_color(0.0f);

  if (task.trace_status == OPTIX_TRACE_STATUS_OPTIONAL_UNUSED)
    return splat_color(1.0f);

  visibility = mul_color(visibility, volume_integrate_transmittance(task.volume_type, task.origin, task.ray, task.limit));

  return visibility;
#else  /* !DIRECT_LIGHTING_NO_SHADOW */
  return splat_color(1.0f);
#endif /* DIRECT_LIGHTING_NO_SHADOW */
}

#endif /* OPTIX_KERNEL */

#endif /* CU_LUMINARY_LIGHT_SHADOW_H */
