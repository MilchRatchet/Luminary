#ifndef CU_BVH_UTILS_H
#define CU_BVH_UTILS_H

#include "memory.cuh"
#include "utils.cuh"

/*
 * Performs alpha test on traversal triangle
 * @param t Triangle to test.
 * @param t_id ID of tested triangle.
 * @param coords Hit coordinates in triangle.
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
LUM_DEVICE_FUNC int bvh_triangle_intersection_alpha_test(TraversalTriangle t, uint32_t t_id, float2 coords) {
  if (t.albedo_tex == TEXTURE_NONE)
    return 0;

  const UV tex_coords = load_triangle_tex_coords(t_id, coords);
  const float4 albedo = tex2D<float4>(device.ptrs.albedo_atlas[t.albedo_tex].tex, tex_coords.u, 1.0f - tex_coords.v);

  if (albedo.w <= device.scene.material.alpha_cutoff) {
    return 2;
  }

  if (albedo.w < 1.0f) {
    return 1;
  }

  return 0;
}

#endif /* CU_BVH_UTILS_H */
