#ifndef CU_TOY_UTILS_H
#define CU_TOY_UTILS_H

#include "math.cuh"
#include "utils.cuh"

__device__ float toy_get_ambient_index_of_refraction(const vec3 position) {
  if (device.scene.ocean.active && position.y < device.scene.ocean.height)
    return device.scene.ocean.refractive_index;

  return 1.0f;
}

/*
 * Requirement:
 *      - Position should be the middle of the shape
 *      - Scale should be the radius of the shape
 */

__device__ float toy_plane_distance(const vec3 origin, const vec3 ray) {
  const vec3 n = rotate_vector_by_quaternion(get_vector(0.0f, 1.0f, 0.0f), device.scene.toy.computed_rotation);

  const float denom = dot_product(n, ray);
  if (fabsf(denom) > eps) {
    const vec3 d  = sub_vector(device.scene.toy.position, origin);
    const float t = dot_product(d, n) / denom;

    if (t <= 0.0f)
      return FLT_MAX;

    const vec3 p  = add_vector(origin, scale_vector(ray, t));
    const float r = get_length(sub_vector(device.scene.toy.position, p));

    if (r >= device.scene.toy.scale)
      return FLT_MAX;

    return t;
  }

  return FLT_MAX;
}

__device__ float get_toy_distance(const vec3 origin, const vec3 ray) {
  switch (device.scene.toy.shape) {
    case TOY_SPHERE:
      return sphere_ray_intersection(ray, origin, device.scene.toy.position, device.scene.toy.scale);
    case TOY_PLANE:
      return toy_plane_distance(origin, ray);
  }

  return FLT_MAX;
}

__device__ vec3 toy_sphere_normal(const vec3 position) {
  const vec3 diff = sub_vector(position, device.scene.toy.position);
  return normalize_vector(diff);
}

__device__ vec3 toy_plane_normal(const vec3 position) {
  return rotate_vector_by_quaternion(get_vector(0.0f, 1.0f, 0.0f), device.scene.toy.computed_rotation);
}

__device__ vec3 get_toy_normal(const vec3 position) {
  switch (device.scene.toy.shape) {
    case TOY_SPHERE:
      return toy_sphere_normal(position);
    case TOY_PLANE:
      return toy_plane_normal(position);
  }

  return get_vector(0.0f, 1.0f, 0.0f);
}

__device__ bool toy_sphere_is_inside(const vec3 position) {
  const vec3 diff = sub_vector(position, device.scene.toy.position);
  return (dot_product(diff, diff) < device.scene.toy.scale * device.scene.toy.scale);
}

__device__ bool toy_plane_is_inside(const vec3 position) {
  const vec3 n = toy_plane_normal(position);

  const float dot = dot_product(n, sub_vector(position, device.scene.toy.position));

  return (dot < 0.0f);
}

__device__ bool toy_is_inside(const vec3 position) {
  switch (device.scene.toy.shape) {
    case TOY_SPHERE:
      return toy_sphere_is_inside(position);
    case TOY_PLANE:
      return toy_plane_is_inside(position);
  }

  return false;
}

__device__ float toy_plane_solid_angle(const vec3 position) {
  // this is not correct, fix this in the future if this is important
  const float sphere = sample_sphere_solid_angle(device.scene.toy.position, device.scene.toy.scale, position);

  const vec3 n = rotate_vector_by_quaternion(get_vector(0.0f, 1.0f, 0.0f), device.scene.toy.computed_rotation);

  return sphere * fabsf(dot_product(n, normalize_vector(sub_vector(device.scene.toy.position, position))));
}

__device__ float toy_get_solid_angle(const vec3 position) {
  switch (device.scene.toy.shape) {
    case TOY_SPHERE:
      return sample_sphere_solid_angle(device.scene.toy.position, device.scene.toy.scale, position);
    case TOY_PLANE:
      return toy_plane_solid_angle(position);
  }

  return 0.0f;
}

__device__ vec3 toy_plane_sample_ray(const vec3 position, const float2 random) {
  const float alpha = random.x * 2.0f * PI;
  const float beta  = sqrtf(random.y);

  const vec3 n = rotate_vector_by_quaternion(get_vector(0.0f, 1.0f, 0.0f), device.scene.toy.computed_rotation);

  const vec3 d = sample_hemisphere_basis(0.0f, alpha, n);
  const vec3 p = add_vector(device.scene.toy.position, scale_vector(d, beta * device.scene.toy.scale));

  return normalize_vector(sub_vector(p, position));
}

__device__ vec3 toy_sample_ray(const vec3 position, const float2 random) {
  switch (device.scene.toy.shape) {
    case TOY_SPHERE:
      float dummy;
      return sample_sphere(device.scene.toy.position, device.scene.toy.scale, position, random, dummy);
    case TOY_PLANE:
      return toy_plane_sample_ray(position, random);
  }

  return normalize_vector(sub_vector(device.scene.toy.position, position));
}

#endif /* CU_TOY_UTILS_H */
