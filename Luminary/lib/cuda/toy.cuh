#ifndef CU_TOY_H
#define CU_TOY_H

#include "math.cuh"

__device__ float toy_sphere_distance(const vec3 origin, const vec3 ray) {
  const vec3 a      = sub_vector(origin, device_scene.toy.position);
  const float u     = dot_product(a, ray);
  const float v     = dot_product(a, a);
  const float b     = u * u;
  const float c     = v - device_scene.toy.scale * device_scene.toy.scale;
  const float delta = b - c;

  if (delta < 0.0f)
    return FLT_MAX;

  const float d = sqrtf(delta);

  float distance1 = -u + d;
  float distance2 = -u - d;

  distance1 = (distance1 < 0.0f) ? FLT_MAX : distance1;
  distance2 = (distance2 < 0.0f) ? FLT_MAX : distance2;

  return fminf(distance1, distance2);
}

__device__ float get_toy_distance(const vec3 origin, const vec3 ray) {
  switch (device_scene.toy.shape) {
    case TOY_SPHERE:
      return toy_sphere_distance(origin, ray);
  }

  return FLT_MAX;
}

__device__ vec3 toy_sphere_normal(const vec3 position) {
  return normalize_vector(sub_vector(position, device_scene.toy.position));
}

__device__ vec3 get_toy_normal(const vec3 position) {
  switch (device_scene.toy.shape) {
    case TOY_SPHERE:
      return toy_sphere_normal(position);
  }

  return normalize_vector(position);
}

#endif /* CU_TOY_H */
