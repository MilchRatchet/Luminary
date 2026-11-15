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

LUMINARY_FUNCTION MaterialContextParticle
  particle_get_context(const DeviceTask task, const DeviceTaskMediumStack& medium, const uint32_t instance_id) {
  const float4 sphere = __ldg(device.ptrs.particle_spheres + (instance_id & HIT_TYPE_PARTICLE_MASK));

  const float time         = random_1D_consistent(RANDOM_TARGET_CAMERA_TIME, task.path_id);
  const vec3 motion        = angles_to_direction(device.particles.direction_altitude, device.particles.direction_azimuth);
  const vec3 motion_offset = scale_vector(motion, time * device.particles.speed);
  const vec3 hit_tile      = scale_vector(add_vector(task.origin, motion_offset), 1.0f / device.particles.scale);

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
  ctx.V           = scale_vector(task.ray, -1.0f);
  ctx.state       = task.state;
  ctx.volume_type = (VolumeType) volume_id;

  return ctx;
}

#endif /* CU_PARTICLE_UTILS_H */
