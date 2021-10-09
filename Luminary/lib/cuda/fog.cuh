#ifndef CU_FOG_H
#define CU_FOG_H

#include "math.cuh"

__device__ float get_intersection_fog(vec3 origin, vec3 ray, float random) {
  float height = device_scene.fog.height + device_scene.fog.falloff;
  float dist   = (fabsf(ray.y) < eps) ? FLT_MAX : (height - origin.y) / ray.y;

  float max_dist = FLT_MAX;
  float min_dist = 0.0f;

  if (dist < 0.0f) {
    if (origin.y > height) {
      max_dist = -FLT_MAX;
    }
    else {
      max_dist = FLT_MAX;
    }
  }
  else {
    if (origin.y > height) {
      min_dist = dist;
    }
    else {
      max_dist = dist;
    }
  }

  max_dist = fminf(device_scene.fog.dist, max_dist);

  float t = logf(random) / (-device_scene.fog.scattering_coeff * 0.00001f);

  return (t < min_dist || t > max_dist) ? FLT_MAX : t;
}

#endif /* CU_FOG_H */
