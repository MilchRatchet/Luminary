#ifndef CU_TRACE_H
#define CU_TRACE_H

#include "memory.cuh"
#include "ocean_utils.cuh"
#include "utils.cuh"

__device__ float trace_preprocess(const DeviceTask task, TriangleHandle& handle_result) {
  const uint32_t pixel = get_pixel_id(task.index);

  float depth   = FLT_MAX;
  handle_result = triangle_handle_get(HIT_TYPE_SKY, 0);

  if (device.ocean.active) {
    if (task.origin.y < OCEAN_MIN_HEIGHT || task.origin.y > OCEAN_MAX_HEIGHT) {
      const float far_distance = ocean_far_distance(task.origin, task.ray);

      if (far_distance < depth) {
        depth                     = far_distance;
        handle_result.instance_id = HIT_TYPE_REJECT;
      }
    }
  }

  return depth;
}

#endif /* CU_TRACE_H */
