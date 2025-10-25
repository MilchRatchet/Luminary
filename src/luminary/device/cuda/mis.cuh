#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
#include "material.cuh"
#include "utils.cuh"

#define MIS_FORCE_FULL_GI (-1.0f)

template <MaterialType TYPE>
LUMINARY_FUNCTION float mis_compute_gi_pdf(const MaterialContext<TYPE> ctx, const vec3 L, const bool is_refraction) {
  return 0.0f;
}

template <>
LUMINARY_FUNCTION float mis_compute_gi_pdf(const MaterialContextGeometry ctx, const vec3 L, const bool is_refraction) {
  float gi_pdf;
  if (is_refraction) {
    const float refraction_index = material_get_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx);

    const vec3 H = bsdf_normal_from_pair(L, ctx.V, refraction_index);

    const float NdotH = dot_product(ctx.normal, H);
    const float NdotV = dot_product(ctx.normal, ctx.V);
    const float NdotL = dot_product(ctx.normal, L);
    const float HdotV = dot_product(H, ctx.V);
    const float HdotL = dot_product(H, L);

    const float roughness  = material_get_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx);
    const float roughness2 = roughness * roughness;
    const float roughness4 = roughness2 * roughness2;

    gi_pdf = bsdf_microfacet_refraction_pdf(ctx, roughness4, NdotH, NdotV, NdotL, HdotV, HdotL, refraction_index);
  }
  else {
    const vec3 H = normalize_vector(add_vector(L, ctx.V));

    const float NdotH = dot_product(ctx.normal, H);
    const float NdotV = dot_product(ctx.normal, ctx.V);

    gi_pdf = bsdf_microfacet_pdf(ctx, NdotH, NdotV);
  }

  return gi_pdf;
}

LUMINARY_FUNCTION float mis_compute_weight_base(const float gi_pdf, const float solid_angle, const float power, const float dist_sq) {
  const float dl_pdf =
    LIGHT_GEO_MAX_SAMPLES * (1.0f / solid_angle) * (power / dist_sq) * (1.0f / device.ptrs.light_scene_data->total_power);

  return gi_pdf / (gi_pdf + dl_pdf);
}

LUMINARY_FUNCTION float mis_compute_weight_gi(const float gi_pdf, const float solid_angle, const float power, const float dist_sq) {
  if (gi_pdf == MIS_FORCE_FULL_GI)
    return 1.0f;

  return mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContext<TYPE> ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angl,
  const bool is_refractione) {
  return 1.0f;
}

template <>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContextGeometry ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle,
  const bool is_refraction) {
  if (device.state.depth == device.settings.max_ray_depth)
    return 1.0f;

  const float light_area = get_length(cross_product(light.edge1, light.edge2)) * 0.5f;
  const float power      = color_importance(light_color) * light_area;
  const vec3 light_center =
    add_vector(light.vertex, add_vector(scale_vector(light.edge1, 1.0f / 3.0f), scale_vector(light.edge2, 1.0f / 3.0f)));
  const vec3 diff_to_center = sub_vector(ctx.position, light_center);
  const float dist_sq       = dot_product(diff_to_center, diff_to_center);
  const float gi_pdf        = mis_compute_gi_pdf(ctx, L, is_refraction);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq);
}

LUMINARY_FUNCTION MISPayload mis_get_payload(const MaterialContextGeometry ctx, const vec3 L, const bool is_refraction) {
  MISPayload mis_data;
  mis_data.origin               = ctx.position;
  mis_data.sampling_probability = mis_compute_gi_pdf(ctx, L, is_refraction);

  return mis_data;
}

#endif /* CU_LUMINARY_MIS_H */
