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

    // Remove entries from the cache point as long as we stay above the importance threshold
    // TODO: Ideally, we remove them in pairs because there is no point in having an odd number of entries
    while (min_heap_remove_min(heap, sum_importance * args.importance_threshold)) {
    }

    // For optimal alignment, each cache point must hold a multiple of 2 entries
    heap.num_elements = ((heap.num_elements + 1) >> 1) << 1;

    args.dst_cache_point_num_entries[id] = heap.num_elements;

    // TODO: Perform reduction to obtain offsets into memory buffer for each cache point

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_LUMINARY_LIGHT_GRID_H */
