#ifndef CU_RIS_H
#define CU_RIS_H

#include "light.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

#if defined(SHADING_KERNEL) && !defined(VOLUME_KERNEL)

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Bit20]
// B. Bitterli, C. Wyman, M. Pharr, P. Shirley, A. Lefohn, W. Jarosz,
// "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting",
// ACM Transactions on Graphics (Proceedings of SIGGRAPH), 39(4), 2020.

// [Wym21]
// C. Wyman, A. Panteleev, "Rearchitecting Spatiotemporal Resampling for Production",
// High-Performance Graphics - Symposium Papers, pp. 23-41, 2021

__device__ TriangleHandle ris_sample_light(
  const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, const uint32_t bsdf_sample_light_key,
  const vec3 bsdf_sample_ray, const bool initial_is_refraction, vec3& selected_ray, RGBF& selected_light_color, float& selected_dist,
  bool& selected_is_refraction) {
  TriangleHandle selected_handle = triangle_handle_get(LIGHT_ID_NONE, 0);

  float sum_weight = 0.0f;

  selected_ray           = get_vector(0.0f, 0.0f, 1.0f);
  selected_light_color   = get_color(0.0f, 0.0f, 0.0f);
  selected_dist          = 1.0f;
  selected_is_refraction = false;

  // TODO: Once the light tree is implemented. Consider reducing the number of samples for deep bounces.
  const int reservoir_size = device.settings.light_num_ris_samples;

  // Don't allow triangles to sample themselves.
  const TriangleHandle blocked_handle = triangle_handle_get(data.instance_id, data.tri_id);

  ////////////////////////////////////////////////////////////////////
  // Initialize reservoir with an initial sample
  ////////////////////////////////////////////////////////////////////

  TriangleHandle bsdf_sample_handle = blocked_handle;
  if (bsdf_sample_light_key != HIT_TYPE_LIGHT_BSDF_HINT) {
    bsdf_sample_handle = device.ptrs.light_tree_tri_handle_map[bsdf_sample_light_key];
  }

  if (!triangle_handle_equal(bsdf_sample_handle, blocked_handle)) {
    const DeviceTransform trans = load_transform(bsdf_sample_handle.instance_id);

    float dist;
    const TriangleLight triangle_light = light_load(bsdf_sample_handle, data.position, bsdf_sample_ray, trans, dist);
    const float solid_angle            = light_get_solid_angle(triangle_light, data.position);

    if (dist < FLT_MAX && solid_angle > 0.0f) {
      RGBF light_color = light_get_color(triangle_light);

      bool is_refraction;
      const RGBF bsdf_weight = bsdf_evaluate(data, bsdf_sample_ray, BSDF_SAMPLING_GENERAL, is_refraction);
      light_color            = mul_color(light_color, bsdf_weight);

      const float target_pdf = color_importance(light_color);

      const float bsdf_sample_pdf = bsdf_sample_for_light_pdf(data, bsdf_sample_ray);

      const float nee_light_tree_pdf      = light_tree_query_pdf(data, bsdf_sample_light_key);
      const float one_over_nee_sample_pdf = solid_angle / (nee_light_tree_pdf * reservoir_size);

      // MIS weight pre multiplied with inverse of pdf, little trick by using inverse of NEE pdf, this is fine because NEE pdf is never 0.
      const float mis_weight = (reservoir_size > 0)
                                 ? bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf
                                     / (bsdf_sample_pdf * bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf + 1.0f)
                                 : 1.0f / bsdf_sample_pdf;

      const float weight = target_pdf * mis_weight;

      sum_weight += weight;

      selected_handle        = bsdf_sample_handle;
      selected_light_color   = (target_pdf > 0.0f) ? scale_color(light_color, 1.0f / target_pdf) : get_color(0.0f, 0.0f, 0.0f);
      selected_ray           = bsdf_sample_ray;
      selected_dist          = dist;
      selected_is_refraction = initial_is_refraction;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Resample NEE samples
  ////////////////////////////////////////////////////////////////////

  float resampling_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_RESAMPLING + light_ray_index, pixel);

  for (int i = 0; i < reservoir_size; i++) {
    const uint32_t random_offset = light_ray_index * reservoir_size + i;

    const float light_tree_random = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_LIGHT_TREE + random_offset, pixel);

    float light_tree_pdf;
    DeviceTransform trans;
    const TriangleHandle light_handle = light_tree_query(data, light_tree_random, light_tree_pdf, trans);

    if (triangle_handle_equal(light_handle, blocked_handle))
      continue;

    uint3 light_uv_packed;
    TriangleLight triangle_light = light_load_sample_init(light_handle, trans, light_uv_packed);

    const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_RAY_DIR + light_ray_index * reservoir_size + i, pixel);

    vec3 ray;
    float dist, solid_angle;
    light_load_sample_finalize(triangle_light, light_uv_packed, data.position, ray_random, ray, dist, solid_angle);

    if (dist == FLT_MAX || solid_angle == 0.0f)
      continue;

    RGBF light_color = light_get_color(triangle_light);

    bool is_refraction;
    const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_refraction);
    light_color            = mul_color(light_color, bsdf_weight);
    const float target_pdf = color_importance(light_color);

    if (target_pdf == 0.0f)
      continue;

    const float bsdf_sample_pdf         = bsdf_sample_for_light_pdf(data, ray);
    const float one_over_nee_sample_pdf = solid_angle / ((float) reservoir_size * light_tree_pdf);

    const float mis_weight =
      one_over_nee_sample_pdf / (bsdf_sample_pdf * bsdf_sample_pdf * one_over_nee_sample_pdf * one_over_nee_sample_pdf + 1.0f);

    const float weight = target_pdf * mis_weight;

    sum_weight += weight;

    const float resampling_probability = weight / sum_weight;

    if (resampling_random < resampling_probability) {
      selected_handle        = light_handle;
      selected_light_color   = scale_color(light_color, 1.0f / target_pdf);
      selected_ray           = ray;
      selected_dist          = dist;
      selected_is_refraction = is_refraction;

      resampling_random = resampling_random / resampling_probability;
    }
    else {
      resampling_random = (resampling_random - resampling_probability) / (1.0f - resampling_probability);
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  ////////////////////////////////////////////////////////////////////

  // Selected light color already includes 1 / target_pdf.
  selected_light_color = scale_color(selected_light_color, sum_weight);

  UTILS_CHECK_NANS(pixel, selected_light_color);

  return selected_handle;
}
#endif /* SHADING_KERNEL && !VOLUME_KERNEL */

#endif /* CU_RIS_H */
