#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
#include "material.cuh"
#include "utils.cuh"

#define MIS_FORCE_FULL_GI (-1.0f)

template <MaterialType TYPE>
__device__ float mis_compute_gi_pdf(const MaterialContext<TYPE> ctx, const vec3 L, const bool is_refraction) {
  return 0.0f;
}

template <>
__device__ float mis_compute_gi_pdf(const MaterialContextGeometry ctx, const vec3 L, const bool is_refraction) {
  float gi_pdf;
  if (is_refraction) {
    const float refraction_index = ctx.ior_in / ctx.ior_out;

    const vec3 H = bsdf_normal_from_pair(L, ctx.V, refraction_index);

    const float NdotH = dot_product(ctx.normal, H);
    const float NdotV = dot_product(ctx.normal, ctx.V);
    const float NdotL = dot_product(ctx.normal, L);
    const float HdotV = dot_product(H, ctx.V);
    const float HdotL = dot_product(H, L);

    gi_pdf = bsdf_microfacet_refraction_pdf(ctx, NdotH, NdotV, NdotL, HdotV, HdotL, refraction_index);
  }
  else {
    const vec3 H = normalize_vector(add_vector(L, ctx.V));

    const float NdotH = dot_product(ctx.normal, H);
    const float NdotV = dot_product(ctx.normal, ctx.V);

    gi_pdf = bsdf_microfacet_pdf(ctx, NdotH, NdotV);
  }

  return gi_pdf;
}

__device__ float mis_compute_weight_base(const float gi_pdf, const float solid_angle, const float power) {
  const float dl_pdf = (1.0f / solid_angle) * power / device.ptrs.light_scene_data->total_power;

  return gi_pdf / (gi_pdf + dl_pdf);
}

__device__ float mis_compute_weight_gi(const float gi_pdf, const float solid_angle, const float power) {
  if (gi_pdf == MIS_FORCE_FULL_GI)
    return 1.0f;

  return mis_compute_weight_base(gi_pdf, solid_angle, power);
}

template <MaterialType TYPE>
__device__ float mis_compute_weight_dl(
  const MaterialContext<TYPE> ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angl,
  const bool is_refractione) {
  return 1.0f;
}

template <>
__device__ float mis_compute_weight_dl(
  const MaterialContextGeometry ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle,
  const bool is_refraction) {
  if (device.state.depth == device.settings.max_ray_depth)
    return 1.0f;

  const float light_area = get_length(cross_product(light.edge1, light.edge2)) * 0.5f;
  const float power      = color_importance(light_color) * light_area;
  const float gi_pdf     = mis_compute_gi_pdf(ctx, L, is_refraction);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power);
}

__device__ DeviceMISPayload mis_get_payload(const MaterialContextGeometry ctx, const vec3 L, const bool is_refraction) {
  DeviceMISPayload mis_data;
  mis_data.origin               = ctx.position;
  mis_data.sampling_probability = mis_compute_gi_pdf(ctx, L, is_refraction);

  return mis_data;
}

#endif /* CU_LUMINARY_MIS_H */
