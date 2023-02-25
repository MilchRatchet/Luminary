#ifndef CLOUD_SHADOW_CUH
#define CLOUD_SHADOW_CUH

#include "cloud_utils.cuh"
#include "math.cuh"
#include "utils.cuh"

__device__ float cloud_shadow(const vec3 origin, const vec3 ray) {
  if (!device.scene.sky.cloud.active) {
    return 1.0f;
  }

  const float2 cloud_layer_intersect = cloud_get_intersection(origin, ray, FLT_MAX);

  const float start = cloud_layer_intersect.x;
  const float dist  = fminf(cloud_layer_intersect.y, 30.0f);

  if (start == FLT_MAX || dist <= 0.0f) {
    return 1.0f;
  }

  const int step_count = device.scene.sky.cloud.steps / 3;

  const int big_step_mult = 2;
  const float big_step    = big_step_mult;

  const float step_size = dist / step_count;

  float reach = start + (white_noise() + 0.1f) * step_size;

  float transmittance = 1.0f;

  for (int i = 0; i < step_count; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height = cloud_height(pos);

    if (height < 0.0f || height > 1.0f) {
      break;
    }

    const vec3 weather = cloud_weather(pos, height);

    if (weather.x < 0.05f) {
      i += big_step_mult - 1;
      reach += step_size * big_step;
      continue;
    }

    const float density = cloud_density(pos, height, weather);

    if (density > 0.0f) {
      transmittance *= expf(-density * CLOUD_EXTINCTION_DENSITY * step_size);
      if (transmittance < 0.005f) {
        transmittance = 0.0f;
        break;
      }
    }
    reach += step_size;
  }

  return transmittance;
}

#endif /* CLOUD_SHADOW_CUH */
