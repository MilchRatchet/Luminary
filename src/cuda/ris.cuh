#ifndef CU_RIS_H
#define CU_RIS_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

__device__ uint32_t ris_sample_light(const GBufferData data, const ushort2 pixel, const uint32_t light_ray_index, float& pdf) {
  const vec3 sky_pos = world_to_sky_transform(data.position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  uint32_t sampled_id = LIGHT_ID_NONE;

  float sum_weight = 0.0f;

  if (sun_visible) {
    sampled_id = LIGHT_ID_SUN;
    sum_weight += 1.0f;
  }

  if (toy_visible) {
    sum_weight += 1.0f;

    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_2 + light_ray_index, pixel) < 1.0f / sum_weight) {
      sampled_id = LIGHT_ID_TOY;
    }
  }

  if (device.scene.material.lights_active) {
    uint32_t id =
      (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_0 + light_ray_index, pixel) * device.scene.triangle_lights_count) + 0.499f;

    float weight = device.scene.triangle_lights_count;

    sum_weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_TBD_3 + light_ray_index, pixel) < weight / sum_weight) {
      sampled_id = id;
    }
  }

  pdf = (sampled_id != LIGHT_ID_NONE) ? 1.0f / sum_weight : 0.0f;

  // TODO: Implement RIS
  return sampled_id;
}

#endif /* CU_RIS_H */
