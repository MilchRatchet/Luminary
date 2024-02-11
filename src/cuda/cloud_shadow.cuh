#ifndef CLOUD_SHADOW_CUH
#define CLOUD_SHADOW_CUH

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky_defines.h"
#include "utils.cuh"

LUM_DEVICE_FUNC bool cloud_shadow_layer(const vec3 origin, const vec3 ray, const int step_count, const CloudLayerType layer) {
  float2 cloud_layer_intersect;
  float max_dist;

  switch (layer) {
    case CLOUD_LAYER_LOW: {
      cloud_layer_intersect = cloud_get_lowlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.scene.sky.cloud.low.height_max - device.scene.sky.cloud.low.height_min);
    } break;
    case CLOUD_LAYER_MID: {
      cloud_layer_intersect = cloud_get_midlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.scene.sky.cloud.mid.height_max - device.scene.sky.cloud.mid.height_min);
    } break;
    case CLOUD_LAYER_TOP: {
      cloud_layer_intersect = cloud_get_toplayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.scene.sky.cloud.top.height_max - device.scene.sky.cloud.top.height_min);
    } break;
    default:
      return false;
  }

  const float start = cloud_layer_intersect.x;
  const float dist  = fminf(cloud_layer_intersect.y, max_dist);

  if (start != FLT_MAX && dist > 0.0f) {
    const float step_size = dist / step_count;

    float reach = start + 0.1f * step_size;

    for (int i = 0; i < step_count; i++) {
      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float height = cloud_height(pos, layer);

      if (height < 0.0f || height > 1.0f) {
        break;
      }

      const CloudWeather weather = cloud_weather(pos, height, layer);

      if (cloud_significant_point(height, weather, layer)) {
        if (cloud_density(pos, height, weather, 2.0f, layer) > 0.0f) {
          return true;
        }
      }

      reach += step_size;
    }
  }

  return false;
}

LUM_DEVICE_FUNC float cloud_shadow(const vec3 origin, const vec3 ray) {
  if (!device.scene.sky.cloud.active || !device.scene.sky.cloud.atmosphere_scattering) {
    return 1.0f;
  }

  if (device.scene.sky.cloud.low.active) {
    if (cloud_shadow_layer(origin, ray, device.scene.sky.cloud.steps / 3, CLOUD_LAYER_LOW)) {
      return 0.0f;
    }
  }

  if (device.scene.sky.cloud.mid.active) {
    if (cloud_shadow_layer(origin, ray, device.scene.sky.cloud.steps / 16, CLOUD_LAYER_MID)) {
      return 0.1f;
    }
  }

  if (device.scene.sky.cloud.top.active) {
    if (cloud_shadow_layer(origin, ray, device.scene.sky.cloud.steps / 32, CLOUD_LAYER_TOP)) {
      return 0.5f;
    }
  }

  return 1.0f;
}

#endif /* CLOUD_SHADOW_CUH */
