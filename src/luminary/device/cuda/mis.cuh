#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
#include "light_bsdf.cuh"
#include "material.cuh"
#include "utils.cuh"

#define MIS_FORCE_FULL_GI (-1.0f)

template <MaterialType TYPE>
LUMINARY_FUNCTION float mis_compute_gi_pdf(const MaterialContext<TYPE> ctx, const vec3 L) {
  return 0.0f;
}

template <>
LUMINARY_FUNCTION float mis_compute_gi_pdf(const MaterialContextGeometry ctx, const vec3 L) {
  return light_bsdf_get_probability(ctx, L);
}

LUMINARY_FUNCTION float mis_compute_weight_base(
  const float gi_pdf, const float solid_angle, const float power, const float dist_sq, const float light_tree_root_sum) {
  const float dl_pdf = LIGHT_GEO_MAX_SAMPLES * (1.0f / solid_angle) * (power / dist_sq) * (1.0f / light_tree_root_sum);

  return gi_pdf / (gi_pdf + dl_pdf);
}

LUMINARY_FUNCTION float mis_compute_weight_gi(
  const float gi_pdf, const float solid_angle, const float power, const float dist_sq, const float light_tree_root_sum) {
  if (gi_pdf == MIS_FORCE_FULL_GI)
    return 1.0f;

  return mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq, light_tree_root_sum);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContext<TYPE> ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle,
  const float light_tree_root_sum) {
  return 1.0f;
}

template <>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContextGeometry ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float solid_angle,
  const float light_tree_root_sum) {
  if (device.state.depth == device.settings.max_ray_depth)
    return 1.0f;

  const float light_area = get_length(cross_product(light.edge1, light.edge2)) * 0.5f;
  const float power      = color_importance(light_color) * light_area;
  const vec3 light_center =
    add_vector(light.vertex, add_vector(scale_vector(light.edge1, 1.0f / 3.0f), scale_vector(light.edge2, 1.0f / 3.0f)));
  const vec3 diff_to_center = sub_vector(ctx.position, light_center);
  const float dist_sq       = dot_product(diff_to_center, diff_to_center);
  const float gi_pdf        = mis_compute_gi_pdf(ctx, L);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq, light_tree_root_sum);
}

LUMINARY_FUNCTION MISPayload
  mis_get_payload(const MaterialContextGeometry ctx, const vec3 L, const bool is_refraction, const float light_tree_root_sum) {
  MISPayload mis_data;
  mis_data.origin               = ctx.position;
  mis_data.light_tree_root_sum  = light_tree_root_sum;
  mis_data.sampling_probability = mis_compute_gi_pdf(ctx, L);

  return mis_data;
}

#endif /* CU_LUMINARY_MIS_H */
