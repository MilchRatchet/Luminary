#ifndef SKY_UTILS_CUH
#define SKY_UTILS_CUH

#include "sky_defines.h"

__device__ float sky_height(const vec3 point) {
  return get_length(point) - SKY_EARTH_RADIUS;
}

__device__ float world_to_sky_scale(float input) {
  return input * 0.001f;
}

__device__ float sky_to_world_scale(float input) {
  return input * 1000.0f;
}

__device__ vec3 world_to_sky_transform(vec3 input) {
  vec3 result;

  result.x = world_to_sky_scale(input.x);
  result.y = world_to_sky_scale(input.y) + SKY_EARTH_RADIUS;
  result.z = world_to_sky_scale(input.z);

  result = add_vector(result, device.scene.sky.geometry_offset);

  return result;
}

__device__ vec3 sky_to_world_transform(vec3 input) {
  vec3 result;

  input = sub_vector(input, device.scene.sky.geometry_offset);

  result.x = sky_to_world_scale(input.x);
  result.y = sky_to_world_scale(input.y - SKY_EARTH_RADIUS);
  result.z = sky_to_world_scale(input.z);

  return result;
}

#endif /* SKY_UTILS_CUH */
