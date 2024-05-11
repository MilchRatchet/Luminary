#ifndef CU_RIS_H
#define CU_RIS_H

#if defined(SHADING_KERNEL)

#include "light.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

__device__ uint32_t ris_sample_light(
  const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, vec3& selected_ray, RGBF& selected_light_color,
  float& selected_dist) {
  uint32_t selected_id = LIGHT_ID_NONE;

  float sum_weight          = 0.0f;
  float selected_target_pdf = 0.0f;

  selected_light_color = get_color(0.0f, 0.0f, 0.0f);

  const uint32_t light_count = (device.scene.material.lights_active) ? (1 << device.restir.light_candidate_pool_size_log2) : 0;

  if (light_count == 0)
    return LIGHT_ID_NONE;

  const int reservoir_size                    = device.restir.initial_reservoir_size;
  const float one_over_reservoir_sampling_pdf = device.scene.triangle_lights_count;

  // Don't allow triangles to sample themselves.
  // TODO: This probably adds biasing.
  uint32_t blocked_light_id = LIGHT_ID_NONE;
  if (data.hit_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
    blocked_light_id = load_triangle_light_id(data.hit_id);
  }

  for (int i = 0; i < reservoir_size; i++) {
    uint32_t presampled_id = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_0 + light_ray_index * reservoir_size + i, pixel) * light_count;
    uint32_t id            = device.ptrs.light_candidates[presampled_id];

    if (id == blocked_light_id)
      continue;

    const TriangleLight triangle_light = load_triangle_light(device.restir.presampled_triangle_lights, presampled_id);

    // Reject if the light has no emission towards us.
    if (device.scene.material.light_side_mode != LIGHT_SIDE_MODE_BOTH) {
      const vec3 face_normal = cross_product(triangle_light.edge1, triangle_light.edge2);
      const float direction  = dot_product(face_normal, sub_vector(triangle_light.vertex, data.position));

      const float side = (device.scene.material.light_side_mode == LIGHT_SIDE_MODE_ONE_CW) ? 1.0f : -1.0f;

      if (direction * side > 0.0f) {
        continue;
      }
    }

    const float2 ray_random = quasirandom_sequence_2D(QUASI_RANDOM_TARGET_TBD_3 + light_ray_index * reservoir_size + i, pixel);

    float solid_angle, dist;
    RGBF light_color;
    const vec3 ray = light_sample_triangle(triangle_light, data, ray_random, solid_angle, dist, light_color);

    const RGBF bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, 1.0f);
    light_color            = mul_color(light_color, bsdf_weight);
    float target_pdf       = luminance(light_color);

    if (isinf(target_pdf) || isnan(target_pdf)) {
      target_pdf = 0.0f;
    }

    // solid_angle == 1 / pdf
    const float weight = target_pdf * solid_angle * one_over_reservoir_sampling_pdf;

    sum_weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_1 + light_ray_index * reservoir_size + i, pixel) * sum_weight < weight) {
      selected_target_pdf  = target_pdf;
      selected_id          = id;
      selected_light_color = light_color;
      selected_ray         = ray;
      selected_dist        = dist;
    }
  }

  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  if (selected_id == LIGHT_ID_NONE) {
    selected_light_color = get_color(0.0f, 0.0f, 0.0f);
  }
  else {
    const float mis_weight         = 1.0f / reservoir_size;
    const float normalization_term = mis_weight * sum_weight;

    selected_light_color = scale_color(selected_light_color, normalization_term / selected_target_pdf);
  }

  return selected_id;
}

#endif /* SHADING_KERNEL */

#endif /* CU_RIS_H */
