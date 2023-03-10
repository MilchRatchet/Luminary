#ifndef CLOUD_SHADOW_CUH
#define CLOUD_SHADOW_CUH

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky_defines.h"
#include "utils.cuh"

__device__ float cloud_shadow(const vec3 origin, const vec3 ray) {
  if (!device.scene.sky.cloud.active || !device.scene.sky.cloud.atmosphere_scattering) {
    return 1.0f;
  }

  if (device.scene.sky.cloud.layer_mid) {
    const float2 cloud_layer_intersect = cloud_get_midlayer_intersection(origin, ray, FLT_MAX);

    const float start    = cloud_layer_intersect.x;
    const float max_dist = 6.0f * world_to_sky_scale(device.scene.sky.cloud.height_mid_max - device.scene.sky.cloud.height_mid_min);
    const float dist     = fminf(cloud_layer_intersect.y, max_dist);

    if (start != FLT_MAX && dist > 0.0f) {
      const int step_count  = device.scene.sky.cloud.steps / 16;
      const float step_size = dist / step_count;

      float reach = start + 0.1f * step_size;

      for (int i = 0; i < step_count; i++) {
        const vec3 pos = add_vector(origin, scale_vector(ray, reach));

        const float height = cloud_height(pos, CLOUD_LAYER_MID);

        if (height < 0.0f || height > 1.0f) {
          break;
        }

        const CloudWeather weather = cloud_weather(pos, height);

        if (cloud_significant_point(height, weather, CLOUD_LAYER_MID)) {
          if (cloud_density(pos, height, weather, 2.0f, CLOUD_LAYER_MID) > 0.0f) {
            return 0.0f;
          }
        }

        reach += step_size;
      }
    }
  }

  if (device.scene.sky.cloud.layer_low) {
    const float2 cloud_layer_intersect = cloud_get_lowlayer_intersection(origin, ray, FLT_MAX);

    const float start    = cloud_layer_intersect.x;
    const float max_dist = 6.0f * world_to_sky_scale(device.scene.sky.cloud.height_low_max - device.scene.sky.cloud.height_low_min);
    const float dist     = fminf(cloud_layer_intersect.y, max_dist);

    if (start != FLT_MAX && dist > 0.0f) {
      const int step_count  = device.scene.sky.cloud.steps / 3;
      const float step_size = dist / step_count;

      float reach = start + 0.1f * step_size;

      for (int i = 0; i < step_count; i++) {
        const vec3 pos = add_vector(origin, scale_vector(ray, reach));

        const float height = cloud_height(pos, CLOUD_LAYER_LOW);

        if (height < 0.0f || height > 1.0f) {
          break;
        }

        const CloudWeather weather = cloud_weather(pos, height);

        if (cloud_significant_point(height, weather, CLOUD_LAYER_LOW)) {
          if (cloud_density(pos, height, weather, 2.0f, CLOUD_LAYER_LOW) > 0.0f) {
            return 0.0f;
          }
        }

        reach += step_size;
      }
    }
  }

  return 1.0f;
}

#endif /* CLOUD_SHADOW_CUH */
