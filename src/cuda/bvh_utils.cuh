#ifndef CU_BVH_UTILS_H
#define CU_BVH_UTILS_H

#include "memory.cuh"
#include "utils.cuh"

enum BVHAlphaResult { BVH_ALPHA_RESULT_OPAQUE = 0, BVH_ALPHA_RESULT_SEMI = 1, BVH_ALPHA_RESULT_TRANSPARENT = 2 } typedef BVHAlphaResult;

/*
 * Performs alpha test on traversal triangle
 * @param t Triangle to test.
 * @param t_id ID of tested triangle.
 * @param coords Hit coordinates in triangle.
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
__device__ BVHAlphaResult bvh_triangle_intersection_alpha_test(TraversalTriangle t, uint32_t t_id, float2 coords) {
  if (t.albedo_tex == TEXTURE_NONE)
    return BVH_ALPHA_RESULT_OPAQUE;

  const UV tex_coords    = load_triangle_tex_coords(t_id, coords);
  const float4 tex_value = tex2D<float4>(device.ptrs.albedo_atlas[t.albedo_tex].tex, tex_coords.u, 1.0f - tex_coords.v);

  if (tex_value.w <= device.scene.material.alpha_cutoff) {
    return BVH_ALPHA_RESULT_TRANSPARENT;
  }

  if (tex_value.w < 1.0f) {
    return BVH_ALPHA_RESULT_SEMI;
  }

  return BVH_ALPHA_RESULT_OPAQUE;
}

#endif /* CU_BVH_UTILS_H */
