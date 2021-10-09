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

  float t = fmaxf(0.0f, min_dist) + logf(random) / (-device_scene.fog.scattering * 0.001f);

  return (t < min_dist || t > max_dist) ? FLT_MAX : t;
}

__device__ float get_fog_depth(float y, float ry, float depth) {
  float height = device_scene.fog.height + device_scene.fog.falloff;

  if (y >= height && ry >= 0.0f)
    return 0.0f;

  if (y < height && ry <= 0.0f)
    return fminf(device_scene.fog.dist, depth);

  if (y < height) {
    return fminf(device_scene.fog.dist, fminf(((height - y) / ry), depth));
  }

  return fmaxf(0.0f, fminf(device_scene.fog.dist, depth - ((height - y) / ry)));
}

__device__ float get_fog_density(float base_density, float height) {
  if (height > device_scene.fog.height) {
    base_density = (height < device_scene.fog.height + device_scene.fog.falloff)
                     ? lerp(base_density, 0.0f, (height - device_scene.fog.height) / device_scene.fog.falloff)
                     : 0.0f;
  }

  return base_density;
}

#endif /* CU_FOG_H */
