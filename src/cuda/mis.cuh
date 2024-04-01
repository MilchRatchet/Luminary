#ifndef CU_MIS_H
#define CU_MIS_H

#include "bsdf.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "restir.cuh"
#include "utils.cuh"

__device__ void mis_reset_data(const int pixel) {
  MISData invalid;
  invalid.light_sampled_technique = -1.0f;
  invalid.bsdf_antagonist_weight  = FLT_MAX;
  invalid.bsdf_data               = 0;

  store_mis_data(invalid, pixel);
}

__device__ MISData mis_gather_data(const BSDFSampleInfo info, const float light_sampled_technique) {
  MISData data;
  data.light_sampled_technique = light_sampled_technique;
  data.bsdf_antagonist_weight  = info.antagonist_weight;
  data.bsdf_data = info.sampled_technique | ((info.is_microfacet_based ? 1u : 0u) << 16) | ((info.is_transparent_pass ? 1u : 0u) << 24);

  return data;
}

__device__ BSDFSampleInfo mis_bsdf_sample_info(const MISData data) {
  BSDFSampleInfo info;

  info.antagonist_weight   = data.bsdf_antagonist_weight;
  info.weight              = get_color(1.0f, 1.0f, 1.0f);
  info.sampled_technique   = (BSDFMaterial) (data.bsdf_data & 0xFF);
  info.is_microfacet_based = ((data.bsdf_data >> 16) & 0xFF);
  info.is_transparent_pass = ((data.bsdf_data >> 24) & 0xFF);

  return info;
}

__device__ bool mis_propagate_data(const GBufferData data, const vec3 ray) {
  return (dot_product(ray, data.V) < 0.99f);
}

__device__ void mis_store_data(const GBufferData data, const RGBF record, const MISData mis_data, const vec3 ray, const int pixel) {
  if (mis_propagate_data(data, ray))
    return;

  store_gbuffer_data(data, pixel);
  device.ptrs.bounce_records_history[pixel] = record;
  store_mis_data(mis_data, pixel);
}

__device__ float mis_weight_light_sampled(const GBufferData data, const vec3 ray, const BSDFSampleInfo info, const float marginal_light) {
  const float marginal_bsdf = bsdf_sample_marginal(data, ray, info);

  return marginal_light / (marginal_light + marginal_bsdf);
}

__device__ float mis_weight_bsdf_sampled(const GBufferData data, const int pixel) {
  const vec3 ray                 = scale_vector(data.V, -1.0f);
  const GBufferData history_data = load_gbuffer_data(pixel);
  const RGBF history_records     = device.ptrs.bounce_records_history[pixel];
  const MISData mis_data         = load_mis_data(pixel);

  const BSDFSampleInfo info = mis_bsdf_sample_info(mis_data);

  const float marginal_bsdf  = bsdf_sample_marginal(history_data, ray, info);
  const float marginal_light = restir_sample_marginal(history_data, history_records, data, mis_data.light_sampled_technique);

  return marginal_bsdf / (marginal_bsdf + marginal_light);
}

#endif /* CU_MIS_H */
