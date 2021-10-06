#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"

__device__ float get_intersection_fog(vec3 origin, vec3 ray, float random) {
  float dist = (fabsf(ray.y) < eps) ? FLT_MAX : (device_scene.fog.height - origin.y) / ray.y;

  float max_dist = FLT_MAX;
  float min_dist = 0.0f;

  if (dist < 0.0f) {
    if (origin.y > device_scene.fog.height) {
      max_dist = -FLT_MAX;
    }
    else {
      max_dist = FLT_MAX;
    }
  }
  else {
    if (origin.y > device_scene.fog.height) {
      min_dist = dist;
    }
    else {
      max_dist = dist;
    }
  }

  float t = random * 100.0f;  // logf(1.0f - random) / (-device_scene.fog.scattering_coeff * 0.00001f);

  return (t < min_dist || t > max_dist) ? FLT_MAX : t;
}

#endif /* CU_FOG_H */
