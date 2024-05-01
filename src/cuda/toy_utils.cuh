#ifndef CU_TOY_UTILS_H
#define CU_TOY_UTILS_H

#include "ior_stack.cuh"
#include "math.cuh"
#include "utils.cuh"

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

#ifdef SHADING_KERNEL

__device__ GBufferData toy_generate_g_buffer(const ToyTask task, const int pixel) {
  vec3 normal = get_toy_normal(task.position);

  if (dot_product(normal, task.ray) > 0.0f) {
    normal = scale_vector(normal, -1.0f);
  }

  uint32_t flags = G_BUFFER_REQUIRES_SAMPLING;

  RGBF emission;
  if (device.scene.toy.emissive) {
    emission = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
  }
  else {
    emission = get_color(0.0f, 0.0f, 0.0f);
  }

  if (toy_is_inside(task.position)) {
    flags |= G_BUFFER_REFRACTION_IS_INSIDE;
  }

  const IORStackMethod ior_stack_method =
    (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior = ior_stack_interact(device.scene.toy.refractive_index, pixel, ior_stack_method);

  GBufferData data;
  data.hit_id             = HIT_TYPE_TOY;
  data.albedo             = device.scene.toy.albedo;
  data.emission           = emission;
  data.normal             = normal;
  data.position           = task.position;
  data.V                  = scale_vector(task.ray, -1.0f);
  data.roughness          = (1.0f - device.scene.toy.material.r);
  data.metallic           = device.scene.toy.material.g;
  data.flags              = flags;
  data.ior_in             = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? device.scene.toy.refractive_index : ray_ior;
  data.ior_out            = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ray_ior : device.scene.toy.refractive_index;
  data.colored_dielectric = 1;

  return data;
}

#endif /* SHADING_KERNEL */

#endif /* CU_TOY_UTILS_H */
