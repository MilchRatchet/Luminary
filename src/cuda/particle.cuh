#include "bench.h"
#include "buffer.h"
#include "math.cuh"
#include "utils.cuh"
#include "utils.h"

LUM_DEVICE_FUNC vec3 particle_transform_relative(vec3 p) {
  return sub_vector(p, device.scene.camera.pos);
}

LUM_DEVICE_FUNC GBufferData particle_generate_g_buffer(const ParticleTask task, const int pixel) {
  Quad q   = load_quad(device.particle_quads, task.hit_id & HIT_TYPE_PARTICLE_MASK);
  q.vertex = particle_transform_relative(q.vertex);
  q.edge1  = particle_transform_relative(q.edge1);
  q.edge2  = particle_transform_relative(q.edge2);

  vec3 normal = (dot_product(task.ray, q.normal) < 0.0f) ? q.normal : scale_vector(q.normal, -1.0f);

  RGBAF albedo;
  albedo.r = device.scene.particles.albedo.r;
  albedo.g = device.scene.particles.albedo.g;
  albedo.b = device.scene.particles.albedo.b;
  albedo.a = 1.0f;

  // Particles BSDF is emulated using volume BSDFs
  uint32_t flags = (!state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) ? G_BUFFER_REQUIRES_SAMPLING : 0;
  flags |= G_BUFFER_VOLUME_HIT;

  GBufferData data;
  data.hit_id    = task.hit_id;
  data.albedo    = albedo;
  data.emission  = get_color(0.0f, 0.0f, 0.0f);
  data.normal    = normal;
  data.position  = task.position;
  data.V         = scale_vector(task.ray, -1.0f);
  data.roughness = 0.0f;
  data.metallic  = 0.0f;
  data.flags     = flags;

  return data;
}
