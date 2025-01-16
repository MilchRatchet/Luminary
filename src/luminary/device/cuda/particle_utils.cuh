#ifndef CU_PARTICLE_UTILS_H
#define CU_PARTICLE_UTILS_H

#include "memory.cuh"
#include "utils.cuh"

__device__ vec3 particle_transform_relative(vec3 p) {
  return sub_vector(p, device.camera.pos);
}

__device__ GBufferData particle_generate_g_buffer(const DeviceTask task, const uint32_t instance_id, const int pixel) {
  Quad q   = load_quad(device.ptrs.particle_quads, instance_id & HIT_TYPE_PARTICLE_MASK);
  q.vertex = particle_transform_relative(q.vertex);
  q.edge1  = particle_transform_relative(q.edge1);
  q.edge2  = particle_transform_relative(q.edge2);

  const vec3 normal = (dot_product(task.ray, q.normal) < 0.0f) ? q.normal : scale_vector(q.normal, -1.0f);

  RGBAF albedo;
  albedo.r = device.particles.albedo.r;
  albedo.g = device.particles.albedo.g;
  albedo.b = device.particles.albedo.b;
  albedo.a = 1.0f;

  const float ray_ior = ior_stack_interact(1.0f, pixel, IOR_STACK_METHOD_PEEK_CURRENT);

  // Particles BSDF is emulated using volume BSDFs
  GBufferData data;
  data.instance_id = instance_id;
  data.tri_id      = 0;
  data.albedo      = albedo;
  data.emission    = get_color(0.0f, 0.0f, 0.0f);
  data.normal      = normal;
  data.position    = task.origin;
  data.V           = scale_vector(task.ray, -1.0f);
  data.roughness   = device.particles.phase_diameter;
  data.state       = task.state;
  data.flags       = 0;
  data.ior_in      = ray_ior;
  data.ior_out     = ray_ior;

  return data;
}

#endif /* CU_PARTICLE_UTILS_H */
