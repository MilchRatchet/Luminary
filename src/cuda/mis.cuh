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
  invalid.bsdf_marginal           = FLT_MAX;

  store_mis_data(invalid, pixel);
}

__device__ bool mis_data_is_invalid(const MISData data) {
  return (data.light_sampled_technique == -1.0f || data.bsdf_marginal == FLT_MAX);
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
  const GBufferData history_data = load_gbuffer_data(pixel);
  const RGBF history_records     = device.ptrs.bounce_records_history[pixel];
  const MISData mis_data         = load_mis_data(pixel);

  // Invalid data means no MIS.
  if (mis_data_is_invalid(mis_data))
    return 1.0f;

  const float marginal_light = restir_sample_marginal(history_data, history_records, data, mis_data.light_sampled_technique);

  return mis_data.bsdf_marginal / (mis_data.bsdf_marginal + marginal_light);
}

#endif /* CU_MIS_H */
