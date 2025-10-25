#ifndef CU_BVH_UTILS_H
#define CU_BVH_UTILS_H

#include "memory.cuh"
#include "texture_utils.cuh"
#include "utils.cuh"

#if 0
enum BVHAlphaResult { BVH_ALPHA_RESULT_OPAQUE = 0, BVH_ALPHA_RESULT_SEMI = 1, BVH_ALPHA_RESULT_TRANSPARENT = 2 } typedef BVHAlphaResult;

/*
 * Performs alpha test on traversal triangle
 * @param t Triangle to test.
 * @param t_id ID of tested triangle.
 * @param coords Hit coordinates in triangle.
 * @result 0 if opaque, 1 if transparent, 2 if alpha cutoff
 */
LUMINARY_FUNCTION BVHAlphaResult bvh_triangle_intersection_alpha_test(TraversalTriangle t, uint32_t t_id, float2 coords) {
  if (t.albedo_tex == TEXTURE_NONE)
    return BVH_ALPHA_RESULT_OPAQUE;

  const UV tex_coords = load_triangle_tex_coords(t_id, coords);
  const float alpha   = texture_load(device.ptrs.albedo_atlas[t.albedo_tex], tex_coords).w;

  if (alpha == 0.0f) {
    return BVH_ALPHA_RESULT_TRANSPARENT;
  }

  if (alpha < 1.0f) {
    return BVH_ALPHA_RESULT_SEMI;
  }

  return BVH_ALPHA_RESULT_OPAQUE;
}
#endif

#endif /* CU_BVH_UTILS_H */
