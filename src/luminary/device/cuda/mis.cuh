#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
#include "material.cuh"
#include "utils.cuh"

__device__ float mis_compute_weight_base(const float gi_pdf, const float solid_angle, const float power) {
  const float dl_pdf = (1.0f / solid_angle) * power / device.ptrs.light_scene_data->total_power;

  return gi_pdf / (gi_pdf + dl_pdf);
}

__device__ float mis_compute_weight_gi(const float gi_pdf, const float solid_angle, const float power) {
  if (gi_pdf < 0.0f)
    return 1.0f;

  return mis_compute_weight_base(gi_pdf, solid_angle, power);
}

template <MaterialType TYPE>
__device__ float mis_compute_weight_dl(
  const MaterialContext<TYPE> ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle) {
  return 1.0f;
}

template <>
__device__ float mis_compute_weight_dl(
  const MaterialContextGeometry ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle) {
  if (device.state.depth == device.settings.max_ray_depth)
    return 1.0f;

  const float light_area = get_length(cross_product(light.edge1, light.edge2)) * 0.5f;
  const float power      = color_importance(light_color) * light_area;

  const vec3 H = normalize_vector(add_vector(L, ctx.V));

  const float NdotH = dot_product(ctx.normal, H);
  const float NdotV = dot_product(ctx.normal, ctx.V);

  const float gi_pdf = bsdf_microfacet_pdf(ctx, NdotH, NdotV);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power);
}

__device__ DeviceMISPayload mis_get_payload(const MaterialContextGeometry ctx, const vec3 L) {
  const vec3 H = normalize_vector(add_vector(L, ctx.V));

  const float NdotH = dot_product(ctx.normal, H);
  const float NdotV = dot_product(ctx.normal, ctx.V);

  const float gi_pdf = bsdf_microfacet_pdf(ctx, NdotH, NdotV);

  DeviceMISPayload mis_data;
  mis_data.origin               = ctx.position;
  mis_data.sampling_probability = gi_pdf;

  return mis_data;
}

#endif /* CU_LUMINARY_MIS_H */
