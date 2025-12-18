#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
#include "light_bsdf.cuh"
#include "material.cuh"
#include "utils.cuh"

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

  return (dl_pdf > 0.0f) ? gi_pdf / (gi_pdf + dl_pdf) : 1.0f;
}

LUMINARY_FUNCTION float mis_compute_weight_gi(
  const vec3 origin, const TriangleLight light, const RGBF light_color, const float dist, const float gi_pdf,
  const float light_tree_root_sum) {
  if (light_tree_root_sum == 0.0f)
    return 1.0f;

  const float light_area  = light_triangle_get_area(light);
  const float solid_angle = light_triangle_get_solid_angle(light, origin);
  const float power       = color_importance(light_color) * light_area;
  const float dist_sq     = dist * dist;

  return mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq, light_tree_root_sum);
}

template <MaterialType TYPE>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContext<TYPE> ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float dist,
  const float solid_angle, const float light_tree_root_sum) {
  return 1.0f;
}

template <>
LUMINARY_FUNCTION float mis_compute_weight_dl(
  const MaterialContextGeometry ctx, const vec3 L, const TriangleLight light, const RGBF light_color, const float dist,
  const float solid_angle, const float light_tree_root_sum) {
  const float light_area = light_triangle_get_area(light);
  const float power      = color_importance(light_color) * light_area;
  const float dist_sq    = dist * dist;
  const float gi_pdf     = mis_compute_gi_pdf(ctx, L);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power, dist_sq, light_tree_root_sum);
}

#endif /* CU_LUMINARY_MIS_H */
