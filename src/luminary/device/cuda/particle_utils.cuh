#ifndef CU_PARTICLE_UTILS_H
#define CU_PARTICLE_UTILS_H

#include "material.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION bool particle_is_hit(const TriangleHandle handle) {
  return PARTICLE_HIT_ID_CHECK(handle.instance_id);
}

LUMINARY_FUNCTION vec3 particle_world_to_tile(const vec3 position, const PathID& path_id) {
  const float time         = random_1D_consistent(RANDOM_TARGET_CAMERA_TIME, path_id) * device.camera.exposure_time;
  const vec3 motion_offset = scale_vector(device.particles.velocity, time);

  return scale_vector(sub_vector(position, motion_offset), device.particles.inv_tile_size);
}

LUMINARY_FUNCTION MaterialContextParticle
  particle_get_context(const DeviceTask task, const uint32_t flags, const DeviceTaskMediumStack& medium, const uint32_t instance_id) {
  const float4 sphere = __ldg(device.ptrs.particle_spheres + (instance_id & HIT_TYPE_PARTICLE_MASK));

  const vec3 hit_tile = particle_world_to_tile(task.origin, task.path_id);

  vec3 delta;
  delta.x = hit_tile.x - sphere.x;
  delta.y = hit_tile.y - sphere.y;
  delta.z = hit_tile.z - sphere.z;

  delta.x -= roundf(delta.x);
  delta.y -= roundf(delta.y);
  delta.z -= roundf(delta.z);

  const vec3 normal = normalize_vector(delta);

  const uint16_t volume_id = medium_stack_volume_peek(medium, false);

  MaterialContextParticle ctx;
  ctx.particle_id = instance_id;
  ctx.position    = task.origin;
  ctx.normal      = normal;
  ctx.V           = neg_vector(task.ray);
  ctx.state       = flags;
  ctx.volume_type = (VolumeType) volume_id;

  return ctx;
}

#endif /* CU_PARTICLE_UTILS_H */
