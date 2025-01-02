#ifndef CU_TRACE_H
#define CU_TRACE_H

#include "memory.cuh"
#include "ocean_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

__device__ float trace_preprocess(const DeviceTask task, TriangleHandle& handle_result) {
  const uint32_t pixel = get_pixel_id(task.index);

  float depth   = FLT_MAX;
  handle_result = triangle_handle_get(HIT_TYPE_SKY, 0);

  // Intersect against the triangle we hit in primary visible in the last frame.
  // This is a heuristic to speed up the BVH traversal.
  // TODO: Implement texture opacity lookup if that will be used in the BVH traversal aswell.
  if (IS_PRIMARY_RAY) {
    const uint32_t pixel        = get_pixel_id(task.index);
    const TriangleHandle handle = triangle_handle_load_history(pixel);

    if (handle.instance_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      const uint32_t mesh_id      = mesh_id_load(handle.instance_id);
      const DeviceTransform trans = load_transform(handle.instance_id);

      const DeviceTriangle* tri_ptr = device.ptrs.triangles[mesh_id];
      const uint32_t triangle_count = device.ptrs.triangle_counts[mesh_id];

      const float4 t0 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 0, 0, handle.tri_id, triangle_count));
      const float4 t1 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 1, 0, handle.tri_id, triangle_count));
      const float4 t2 = __ldg((float4*) triangle_get_entry_address(tri_ptr, 2, 0, handle.tri_id, triangle_count));

      const vec3 vertex = transform_apply(trans, get_vector(t0.x, t0.y, t0.z));
      const vec3 edge1  = transform_apply(trans, get_vector(t0.w, t1.x, t1.y));
      const vec3 edge2  = transform_apply(trans, get_vector(t1.z, t1.w, t2.x));

      float2 coords;
      const float tri_dist = bvh_triangle_intersection(vertex, edge1, edge2, task.origin, task.ray, coords);

      if (tri_dist < depth) {
        depth         = tri_dist;
        handle_result = handle;
      }
    }
  }

  if (device.toy.active) {
    const float toy_dist = get_toy_distance(task.origin, task.ray);

    if (toy_dist < depth) {
      depth                     = toy_dist;
      handle_result.instance_id = HIT_TYPE_TOY;
    }
  }

  if (device.ocean.active) {
    if (task.origin.y < OCEAN_MIN_HEIGHT && task.origin.y > OCEAN_MAX_HEIGHT) {
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
