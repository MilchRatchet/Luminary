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

__device__ RGBF sky_hdri_sample(const vec3 ray, const float mip_bias) {
  const float theta = atan2f(ray.z, ray.x);
  const float phi   = asinf(ray.y);

  const float u = (theta + PI) / (2.0f * PI);
  const float v = 1.0f - ((phi + 0.5f * PI) / PI);

  const float4 hdri = tex2DLod<float4>(device.ptrs.sky_hdri_luts[0], u, v, mip_bias + device.scene.sky.hdri_mip_bias);

  return get_color(hdri.x, hdri.y, hdri.z);
}

#endif /* SKY_UTILS_CUH */
