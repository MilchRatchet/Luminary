#ifndef CU_RIS_H
#define CU_RIS_H

#include "light.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

#if defined(SHADING_KERNEL)

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

__device__ uint32_t ris_sample_light(
  const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, const uint32_t initial_sample_id, const vec3 initial_ray,
  const bool initial_is_refraction, vec3& selected_ray, RGBF& selected_light_color, float& selected_dist, bool& selected_is_refraction) {
  uint32_t selected_id = LIGHT_ID_NONE;

  float sum_weight          = 0.0f;
  float selected_target_pdf = FLT_MAX;

  selected_light_color = get_color(0.0f, 0.0f, 0.0f);

  if (!device.scene.material.lights_active)
    return LIGHT_ID_NONE;

  const int reservoir_size                    = device.ris_settings.initial_reservoir_size;
  const float one_over_reservoir_pdf_and_size = device.scene.triangle_lights_count / ((float) reservoir_size);

  // Don't allow triangles to sample themselves.
  // TODO: This probably adds biasing.
  uint32_t blocked_light_id = LIGHT_ID_NONE;
  if (data.hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
    blocked_light_id = load_triangle_light_id(data.hit_id);
  }

  ////////////////////////////////////////////////////////////////////
  // Initialize reservoir with an initial sample
  ////////////////////////////////////////////////////////////////////

  if (initial_sample_id != HIT_TYPE_LIGHT_BSDF_HINT && initial_sample_id != blocked_light_id) {
    const TriangleLight triangle_light = load_triangle_light(device.scene.triangle_lights, initial_sample_id);

    float solid_angle, dist;
    RGBF light_color;
    light_sample_triangle_presampled(triangle_light, data, initial_ray, solid_angle, dist, light_color);

    bool is_refraction;
    const RGBF bsdf_weight = bsdf_evaluate(data, initial_ray, BSDF_SAMPLING_GENERAL, is_refraction, 1.0f);
    light_color            = mul_color(light_color, bsdf_weight);

    float target_pdf = luminance(light_color);

    if (isinf(target_pdf) || isnan(target_pdf)) {
      target_pdf = 0.0f;
    }

    // TODO: Consider using one of the pair wise MIS if computing MIS turns out to be too expensive.
    const float bsdf_sample_pdf         = bsdf_sample_for_light_pdf(data, initial_ray);
    const float one_over_nee_sample_pdf = solid_angle * one_over_reservoir_pdf_and_size;

    // MIS weight pre multiplied with inverse of pdf, little trick by using inverse of NEE pdf, this is fine because NEE pdf is never 0.
    const float mis_weight = one_over_nee_sample_pdf / (bsdf_sample_pdf * one_over_nee_sample_pdf + 1.0f);

    const float weight = target_pdf * mis_weight;

    sum_weight += weight;

    selected_target_pdf    = target_pdf;
    selected_id            = initial_sample_id;
    selected_light_color   = light_color;
    selected_ray           = initial_ray;
    selected_dist          = dist;
    selected_is_refraction = initial_is_refraction;
  }

  ////////////////////////////////////////////////////////////////////
  // Resample NEE samples
  ////////////////////////////////////////////////////////////////////

  for (int i = 0; i < reservoir_size; i++) {
    uint32_t id_rand = quasirandom_sequence_1D_base(
      QUASI_RANDOM_TARGET_RIS_LIGHT_ID + light_ray_index * reservoir_size + i, pixel, device.temporal_frames, device.depth);
    uint32_t presampled_id = id_rand >> (31 - device.ris_settings.light_candidate_pool_size_log2);
    uint32_t id            = device.ptrs.light_candidates[presampled_id];

    if (id == blocked_light_id)
      continue;

    const TriangleLight triangle_light = load_triangle_light(device.ris_settings.presampled_triangle_lights, presampled_id);

    // Reject if the light has no emission towards us.
    if (device.scene.material.light_side_mode != LIGHT_SIDE_MODE_BOTH) {
      const vec3 face_normal = cross_product(triangle_light.edge1, triangle_light.edge2);
      const float direction  = dot_product(face_normal, sub_vector(triangle_light.vertex, data.position));

      const float side = (device.scene.material.light_side_mode == LIGHT_SIDE_MODE_ONE_CW) ? 1.0f : -1.0f;

      if (direction * side > 0.0f) {
        continue;
      }
    }

    const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_RIS_RAY_DIR + light_ray_index * reservoir_size + i, pixel);

    float solid_angle, dist;
    RGBF light_color;
    const vec3 ray = light_sample_triangle(triangle_light, data, ray_random, solid_angle, dist, light_color);

    bool is_refraction;
    const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_refraction, 1.0f);
    light_color            = mul_color(light_color, bsdf_weight);
    float target_pdf       = luminance(light_color);

    if (isinf(target_pdf) || isnan(target_pdf)) {
      target_pdf = 0.0f;
    }

    const float bsdf_sample_pdf         = bsdf_sample_for_light_pdf(data, ray);
    const float one_over_nee_sample_pdf = solid_angle * one_over_reservoir_pdf_and_size;

    const float mis_weight = one_over_nee_sample_pdf / (bsdf_sample_pdf * one_over_nee_sample_pdf + 1.0f);

    const float weight = target_pdf * mis_weight;

    sum_weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RIS_RESAMPLING + light_ray_index * reservoir_size + i, pixel) * sum_weight < weight) {
      selected_target_pdf    = target_pdf;
      selected_id            = id;
      selected_light_color   = light_color;
      selected_ray           = ray;
      selected_dist          = dist;
      selected_is_refraction = is_refraction;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  ////////////////////////////////////////////////////////////////////

  selected_light_color = scale_color(selected_light_color, sum_weight / selected_target_pdf);

  return selected_id;
}

#else /* SHADING_KERNEL */

LUMINARY_KERNEL void ris_presample_lights() {
  if (device.scene.triangle_lights_count == 0)
    return;

  int id = THREAD_ID;

  const int light_sample_bin_count = 1 << device.ris_settings.light_candidate_pool_size_log2;
  while (id < light_sample_bin_count) {
    const uint32_t sampled_id =
      quasirandom_sequence_1D_base(id, make_ushort2(0, 0), device.temporal_frames, 0) % device.scene.triangle_lights_count;

    device.ptrs.light_candidates[id]                   = sampled_id;
    device.ris_settings.presampled_triangle_lights[id] = device.scene.triangle_lights[sampled_id];

    id += blockDim.x * gridDim.x;
  }
}

#endif /* !SHADING_KERNEL */

#endif /* CU_RIS_H */
