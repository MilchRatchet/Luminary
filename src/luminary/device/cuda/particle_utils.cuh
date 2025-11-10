#ifndef CU_PARTICLE_UTILS_H
#define CU_PARTICLE_UTILS_H

#include "material.cuh"
#include "memory.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION bool particle_is_hit(const TriangleHandle handle) {
  return PARTICLE_HIT_CHECK(handle.instance_id);
}

LUMINARY_FUNCTION MaterialContextParticle particle_get_context(const DeviceTask task, const uint32_t instance_id) {
  const Quad q = load_quad(device.ptrs.particle_quads, instance_id & HIT_TYPE_PARTICLE_MASK);

  const vec3 normal = (dot_product(task.ray, q.normal) < 0.0f) ? q.normal : scale_vector(q.normal, -1.0f);

  MaterialContextParticle ctx;
  ctx.particle_id = instance_id;
  ctx.position    = task.origin;
  ctx.normal      = normal;
  ctx.V           = scale_vector(task.ray, -1.0f);
  ctx.state       = task.state;
  ctx.volume_type = VolumeType(task.volume_id);

  return ctx;
}

#endif /* CU_PARTICLE_UTILS_H */
