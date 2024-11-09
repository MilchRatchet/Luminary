#ifndef CU_TRACE_H
#define CU_TRACE_H

#include "ocean_utils.cuh"
#include "toy_utils.cuh"
#include "utils.cuh"

__device__ float trace_preprocess(const DeviceTask task, uint32_t& instance_id) {
  const uint32_t pixel = get_pixel_id(task.index);

  float depth = FLT_MAX;
  instance_id = HIT_TYPE_SKY;

  // Intersect against the triangle we hit in primary visible in the last frame.
  // This is a heuristic to speed up the BVH traversal.
  // TODO: Reenable this.
#if 0
  if (IS_PRIMARY_RAY) {
    uint32_t t_id;
    TraversalTriangle tt;
    uint32_t material_id;

    t_id = device.ptrs.hit_id_history[get_pixel_id(task.index)];
    if (t_id <= HIT_TYPE_TRIANGLE_ID_LIMIT) {
      material_id = load_triangle_material_id(t_id);

      const float4 data0 = __ldg((float4*) triangle_get_entry_address(0, 0, t_id));
      const float4 data1 = __ldg((float4*) triangle_get_entry_address(1, 0, t_id));
      const float data2  = __ldg((float*) triangle_get_entry_address(2, 0, t_id));

      tt.vertex = get_vector(data0.x, data0.y, data0.z);
      tt.edge1  = get_vector(data0.w, data1.x, data1.y);
      tt.edge2  = get_vector(data1.z, data1.w, data2);
      tt.id     = t_id;

      const DeviceMaterial mat = load_material(device.ptrs.materials, material_id);

      tt.albedo_tex = mat.albedo_tex;

      // This optimization does not work with displacement.
      if (mat.normal_tex == TEXTURE_NONE) {
        float2 coords;
        const float dist = bvh_triangle_intersection_uv(tt, task.origin, task.ray, coords);

        if (dist < depth) {
          const BVHAlphaResult alpha_result = bvh_triangle_intersection_alpha_test(tt, t_id, coords);

          if (alpha_result != BVH_ALPHA_RESULT_TRANSPARENT) {
            depth  = dist;
            hit_id = t_id;
          }
        }
      }
    }
  }
#endif

  if (device.toy.active) {
    const float toy_dist = get_toy_distance(task.origin, task.ray);

    if (toy_dist < depth) {
      depth       = toy_dist;
      instance_id = HIT_TYPE_TOY;
    }
  }

  if (device.ocean.active) {
    if (task.origin.y < OCEAN_MIN_HEIGHT && task.origin.y > OCEAN_MAX_HEIGHT) {
      const float far_distance = ocean_far_distance(task.origin, task.ray);

      if (far_distance < depth) {
        depth       = far_distance;
        instance_id = HIT_TYPE_REJECT;
      }
    }
  }

  return depth;
}

#endif /* CU_TRACE_H */
