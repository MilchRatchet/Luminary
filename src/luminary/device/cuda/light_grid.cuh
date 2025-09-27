#ifndef CU_LUMINARY_LIGHT_GRID_H
#define CU_LUMINARY_LIGHT_GRID_H

#include "light_tree.cuh"
#include "material.cuh"
#include "math.cuh"
#include "min_heap.cuh"
#include "random.cuh"
#include "utils.cuh"

LUMINARY_KERNEL void light_grid_generate(const KernelArgsLightGridGenerate args) {
  uint32_t id = THREAD_ID;

  while (id < args.count) {
    const float3 random_pos = random_3D(id);

    const vec3 pos = add_vector(mul_vector(get_vector(random_pos.x, random_pos.y, random_pos.z), args.bounds_span), args.bounds_min);

    MaterialContextCachePoint ctx;
    ctx.position    = pos;
    ctx.normal      = get_vector(0.0f, 0.0f, 0.0f);
    ctx.directional = false;

    MinHeap heap = min_heap_get(args.min_heap_data + id * args.allocated_entries, args.allocated_entries);

    float max_importance;
    float sum_importance;
    light_tree_build_max_subset(ctx, heap, max_importance, sum_importance);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_LUMINARY_LIGHT_GRID_H */
