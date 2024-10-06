#ifndef CU_TOY_UTILS_H
#define CU_TOY_UTILS_H

#include "ior_stack.cuh"
#include "math.cuh"
#include "utils.cuh"

/*
 * Toy is deprecated and will soon be removed in favor of preset meshlets that we can then instantiate.
 */

__device__ float get_toy_distance(const vec3 origin, const vec3 ray) {
  return sphere_ray_intersection(ray, origin, device.toy.position, device.toy.scale);
}

__device__ vec3 toy_sphere_normal(const vec3 position) {
  const vec3 diff = sub_vector(position, device.toy.position);
  return normalize_vector(diff);
}

__device__ vec3 get_toy_normal(const vec3 position) {
  return toy_sphere_normal(position);
}

__device__ bool toy_is_inside(const vec3 position, const vec3 ray) {
  const float dist = get_length(sub_vector(position, device.toy.position));

  if (fabsf(dist - device.toy.scale) < 32.0f * eps) {
    const vec3 normal = get_toy_normal(position);

    return (dot_product(normal, ray) >= 0.0f);
  }

  return (dist < device.toy.scale);
}

__device__ float toy_get_solid_angle(const vec3 position) {
  return sample_sphere_solid_angle(device.toy.position, device.toy.scale, position);
}

__device__ vec3 toy_sample_ray(const vec3 position, const float2 random) {
  float dummy;
  return sample_sphere(device.toy.position, device.toy.scale, position, random, dummy);
}

#ifdef SHADING_KERNEL

__device__ GBufferData toy_generate_g_buffer(const ShadingTask task, const ShadingTaskAuxData aux_data, const int pixel) {
  vec3 normal = get_toy_normal(task.position);

  if (dot_product(normal, task.ray) > 0.0f) {
    normal = scale_vector(normal, -1.0f);
  }

  uint32_t flags = G_BUFFER_COLORED_DIELECTRIC;

  RGBF emission;
  if (device.toy.emissive) {
    emission = scale_color(device.toy.emission, device.toy.material.b);
  }
  else {
    emission = get_color(0.0f, 0.0f, 0.0f);
  }

  if (toy_is_inside(task.position, task.ray)) {
    flags |= G_BUFFER_REFRACTION_IS_INSIDE;
  }

  const IORStackMethod ior_stack_method =
    (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PEEK_PREVIOUS : IOR_STACK_METHOD_PEEK_CURRENT;
  const float ray_ior = ior_stack_interact(device.toy.refractive_index, pixel, ior_stack_method);

  GBufferData data;
  data.instance_id = HIT_TYPE_TOY;
  data.tri_id      = 0;
  data.albedo      = device.toy.albedo;
  data.emission    = emission;
  data.normal      = normal;
  data.position    = task.position;
  data.V           = scale_vector(task.ray, -1.0f);
  data.roughness   = (1.0f - device.toy.material.r);
  data.metallic    = device.toy.material.g;
  data.state       = aux_data.state;
  data.flags       = flags;
  data.ior_in      = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? device.toy.refractive_index : ray_ior;
  data.ior_out     = (flags & G_BUFFER_REFRACTION_IS_INSIDE) ? ray_ior : device.toy.refractive_index;

  return data;
}

#endif /* SHADING_KERNEL */

#endif /* CU_TOY_UTILS_H */
