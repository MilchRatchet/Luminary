#ifndef CU_RIS_H
#define CU_RIS_H

#include "ltc.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ uint32_t ris_sample_light(const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, float& sampling_weight) {
  const vec3 sky_pos = world_to_sky_transform(data.position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  uint32_t selected_id = LIGHT_ID_NONE;

  float sum_weight          = 0.0f;
  float selected_target_pdf = 0.0f;

  if (sun_visible) {
    selected_id         = LIGHT_ID_SUN;
    selected_target_pdf = sample_sphere_solid_angle(device.sun_pos, SKY_SUN_RADIUS, sky_pos) * 2e+04f * device.scene.sky.sun_strength;
    sum_weight += selected_target_pdf;
  }

  const uint32_t light_count = (device.scene.material.lights_active) ? (1 << device.restir.light_candidate_pool_size_log2) : 0;
  const int reservoir_size   = (device.scene.triangle_lights_count > 0) ? min(device.restir.initial_reservoir_size, light_count) : 0;

  if (toy_visible) {
    const float sampled_target_pdf =
      toy_get_solid_angle(data.position) * luminance(device.scene.toy.emission) * device.scene.toy.material.b;
    sum_weight += sampled_target_pdf;

    if (
      quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_1 + light_ray_index * (reservoir_size + 1), pixel) * sum_weight
      < sampled_target_pdf) {
      selected_target_pdf = sampled_target_pdf;
      selected_id         = LIGHT_ID_TOY;
    }
  }

  const float reservoir_sampling_pdf = (1.0f / device.scene.triangle_lights_count);

  // Don't allow triangles to sample themselves.
  // TODO: This probably adds biasing.
  uint32_t blocked_light_id = LIGHT_ID_TRIANGLE_ID_LIMIT + 1;
  if (data.hit_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) {
    blocked_light_id = load_triangle_light_id(data.hit_id);
  }

  LTCCoefficients coeffs = ltc_get_coefficients(data);

  for (int i = 0; i < reservoir_size; i++) {
    uint32_t presampled_id = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_0 + light_ray_index * reservoir_size + i, pixel) * light_count;
    uint32_t id            = device.ptrs.light_candidates[presampled_id];

    if (id == blocked_light_id)
      continue;

    TriangleLight triangle_light = load_triangle_light(device.restir.presampled_triangle_lights, presampled_id);

    const float sampled_target_pdf = ltc_integrate(data, coeffs, triangle_light) * device.scene.material.default_material.b;
    const float sampled_pdf        = reservoir_sampling_pdf;

    const float weight = (sampled_pdf > 0.0f) ? sampled_target_pdf / sampled_pdf : 0.0f;

    sum_weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_1 + light_ray_index * (reservoir_size + 1) + 1 + i, pixel) * sum_weight < weight) {
      selected_target_pdf = sampled_target_pdf;
      selected_id         = id;
    }
  }

  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  if (selected_id == LIGHT_ID_NONE) {
    sampling_weight = 0.0f;
  }
  else {
    // We use uniform MIS weights because the images of our distributions are a partition of the set of all lights.
    const float mis_weight = (selected_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? 1.0f / reservoir_size : 1.0f;

    const float normalization_term = mis_weight * sum_weight;

    sampling_weight = normalization_term / selected_target_pdf;
  }

  return selected_id;
}

#endif /* CU_RIS_H */
