#ifndef CU_TOY_H
#define CU_TOY_H

#include "math.cuh"
#include "random.cuh"
#include "state.cuh"
#include "toy_utils.cuh"

LUM_DEVICE_FUNC GBufferData toy_generate_g_buffer(const ToyTask task, const int pixel) {
  vec3 normal = get_toy_normal(task.position);

  if (dot_product(normal, task.ray) > 0.0f) {
    normal = scale_vector(normal, -1.0f);
  }

  uint32_t flags = 0;

  if (
    device.iteration_type == TYPE_LIGHT
    || quasirandom_sequence_1D(QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY, pixel) > device.scene.toy.albedo.a) {
    flags |= G_BUFFER_TRANSPARENT_PASS;
  }

  if (!(flags & G_BUFFER_TRANSPARENT_PASS) && !state_peek(pixel, STATE_FLAG_LIGHT_OCCUPIED)) {
    flags |= G_BUFFER_REQUIRES_SAMPLING;
  }

  vec3 pos;
  if (flags & G_BUFFER_TRANSPARENT_PASS) {
    pos = add_vector(task.position, scale_vector(task.ray, 8.0f * eps * get_length(task.position)));
  }
  else {
    pos = add_vector(task.position, scale_vector(task.ray, -8.0f * eps * get_length(task.position)));
  }

  RGBF emission;
  if (device.scene.toy.emissive) {
    emission = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
  }
  else {
    emission = get_color(0.0f, 0.0f, 0.0f);
  }

  GBufferData data;
  data.hit_id    = HIT_TYPE_TOY;
  data.albedo    = device.scene.toy.albedo;
  data.emission  = emission;
  data.normal    = normal;
  data.position  = pos;
  data.V         = scale_vector(task.ray, -1.0f);
  data.roughness = (1.0f - device.scene.toy.material.r);
  data.metallic  = device.scene.toy.material.g;
  data.flags     = flags;

  return data;
}

#endif /* CU_TOY_H */
