#ifndef CU_LUMINARY_MIS_H
#define CU_LUMINARY_MIS_H

#include "bsdf_utils.cuh"
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

__device__ float mis_compute_weight_dl(const GBufferData data, const vec3 L, const float solid_angle, const float power) {
  if (device.state.depth == device.settings.max_ray_depth)
    return 1.0f;

  const vec3 H = normalize_vector(add_vector(L, data.V));

  const float NdotH = dot_product(data.normal, H);
  const float NdotV = dot_product(data.normal, data.V);

  const float gi_pdf = bsdf_microfacet_pdf(data, NdotH, NdotV);

  return 1.0f - mis_compute_weight_base(gi_pdf, solid_angle, power);
}

__device__ DeviceMISData mis_get_payload(const GBufferData data, const vec3 L) {
  const vec3 H = normalize_vector(add_vector(L, data.V));

  const float NdotH = dot_product(data.normal, H);
  const float NdotV = dot_product(data.normal, data.V);

  const float gi_pdf = bsdf_microfacet_pdf(data, NdotH, NdotV);

  DeviceMISData mis_data;
  mis_data.origin               = data.position;
  mis_data.sampling_probability = gi_pdf;

  return mis_data;
}

#endif /* CU_LUMINARY_MIS_H */
