#ifndef CLOUD_SHADOW_CUH
#define CLOUD_SHADOW_CUH

#include "cloud_utils.cuh"
#include "math.cuh"
#include "sky_defines.h"
#include "utils.cuh"

template <CloudLayerType LAYER_TYPE>
LUMINARY_FUNCTION bool cloud_shadow_layer(const vec3 origin, const vec3 ray, const int step_count) {
  float2 cloud_layer_intersect;
  float max_dist;

  switch (LAYER_TYPE) {
    case CLOUD_LAYER_LOW: {
      cloud_layer_intersect = cloud_get_lowlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.cloud.low.height_max - device.cloud.low.height_min);
    } break;
    case CLOUD_LAYER_MID: {
      cloud_layer_intersect = cloud_get_midlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.cloud.mid.height_max - device.cloud.mid.height_min);
    } break;
    case CLOUD_LAYER_TOP: {
      cloud_layer_intersect = cloud_get_toplayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 6.0f * (device.cloud.top.height_max - device.cloud.top.height_min);
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

      const float height = cloud_height<LAYER_TYPE>(pos);

      if (height < 0.0f || height > 1.0f) {
        continue;
      }

      const CloudWeather weather = cloud_weather<LAYER_TYPE>(pos, height);

      if (cloud_significant_point<LAYER_TYPE>(height, weather)) {
        if (cloud_density<LAYER_TYPE>(pos, height, weather, 2.0f) > 0.0f) {
          return true;
        }
      }

      reach += step_size;
    }
  }

  return false;
}

LUMINARY_FUNCTION float cloud_shadow(const vec3 origin, const vec3 ray) {
  if (!device.cloud.active || !device.cloud.atmosphere_scattering) {
    return 1.0f;
  }

  if (device.cloud.low_active) {
    if (cloud_shadow_layer<CLOUD_LAYER_LOW>(origin, ray, device.cloud.steps / 3)) {
      return 0.0f;
    }
  }

  if (device.cloud.mid_active) {
    if (cloud_shadow_layer<CLOUD_LAYER_MID>(origin, ray, device.cloud.steps / 16)) {
      return 0.1f;
    }
  }

  if (device.cloud.top_active) {
    if (cloud_shadow_layer<CLOUD_LAYER_TOP>(origin, ray, device.cloud.steps / 32)) {
      return 0.5f;
    }
  }

  return 1.0f;
}

#endif /* CLOUD_SHADOW_CUH */
