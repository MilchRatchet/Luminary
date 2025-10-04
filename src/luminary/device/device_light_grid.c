#include "device_light_grid.h"

#include <float.h>

#include "device.h"
#include "host_intrinsics.h"
#include "host_math.h"
#include "internal_error.h"
#include "kernel_args.h"

////////////////////////////////////////////////////////////////////
// Light Grid Cache Implementation
////////////////////////////////////////////////////////////////////

struct LightGridCacheInstance {
  uint32_t mesh_id;
  Quaternion rotation;
  vec3 scale;
  vec3 translation;
} typedef LightGridCacheInstance;

struct LightGridCache {
  bool is_dirty;
  Vec128 min_bounds;
  Vec128 max_bounds;
  ARRAY MeshBoundingBox* mesh_bounding_boxes;
  ARRAY LightGridCacheInstance* instances;
} typedef LightGridCache;

static LuminaryResult _light_grid_cache_create(LightGridCache** cache) {
  __CHECK_NULL_ARGUMENT(cache);

  __FAILURE_HANDLE(host_malloc(cache, sizeof(LightGridCache)));
  memset(*cache, 0, sizeof(LightGridCache));

  __FAILURE_HANDLE(array_create(&(*cache)->mesh_bounding_boxes, sizeof(MeshBoundingBox), 16));
  __FAILURE_HANDLE(array_create(&(*cache)->instances, sizeof(LightGridCacheInstance), 16));

  (*cache)->min_bounds = vec128_set(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
  (*cache)->max_bounds = vec128_set(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_update_mesh(LightGridCache* cache, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(mesh);

  uint32_t num_meshes;
  __FAILURE_HANDLE(array_get_num_elements(cache->mesh_bounding_boxes, &num_meshes));

  if (mesh->id >= num_meshes) {
    __FAILURE_HANDLE(array_set_num_elements(&cache->mesh_bounding_boxes, mesh->id + 1));
    cache->is_dirty = true;
  }

  MeshBoundingBox* cache_mesh = cache->mesh_bounding_boxes + mesh->id;

  if (memcmp(cache_mesh, &mesh->bounding_box, sizeof(MeshBoundingBox))) {
    *cache_mesh     = mesh->bounding_box;
    cache->is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_update_instance(LightGridCache* cache, const MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(instance);

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements(cache->instances, &num_instances));

  if (instance->id >= num_instances) {
    __FAILURE_HANDLE(array_set_num_elements(&cache->instances, instance->id + 1));
    cache->is_dirty = true;
  }

  LightGridCacheInstance* cache_instance = cache->instances + instance->id;

  if (cache_instance->mesh_id != instance->mesh_id) {
    cache_instance->mesh_id = instance->mesh_id;
    cache->is_dirty         = true;
  }

  const Quaternion rotation = rotation_euler_angles_to_quaternion(instance->rotation);

  if (memcmp(&cache_instance->rotation, &rotation, sizeof(Quaternion))) {
    cache_instance->rotation = rotation;
    cache->is_dirty          = true;
  }

  if (memcmp(&cache_instance->scale, &instance->scale, sizeof(vec3))) {
    cache_instance->scale = instance->scale;
    cache->is_dirty       = true;
  }

  if (memcmp(&cache_instance->translation, &instance->translation, sizeof(vec3))) {
    cache_instance->translation = instance->translation;
    cache->is_dirty             = true;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_get_scene_bounding_box(LightGridCache* cache, Vec128* min_bounds, Vec128* max_bounds) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(min_bounds);
  __CHECK_NULL_ARGUMENT(max_bounds);

  Vec128 min = vec128_set_1(FLT_MAX);
  Vec128 max = vec128_set_1(-FLT_MAX);

  uint32_t num_instances;
  __FAILURE_HANDLE(array_get_num_elements(cache->instances, &num_instances));

  for (uint32_t instance_id = 0; instance_id < num_instances; instance_id++) {
    const LightGridCacheInstance* instance = cache->instances + instance_id;

    const MeshBoundingBox mesh_bounding_box = cache->mesh_bounding_boxes[instance->mesh_id];

    const Vec128 offset   = vec128_set(instance->translation.x, instance->translation.y, instance->translation.z, 0.0f);
    const Vec128 scale    = vec128_set(instance->scale.x, instance->scale.y, instance->scale.z, 1.0f);
    const Vec128 rotation = vec128_set(-instance->rotation.x, -instance->rotation.y, -instance->rotation.z, instance->rotation.w);

    // TODO: Implement as a shuffle intrinsic
    Vec128 v0 = vec128_set(mesh_bounding_box.min.x, mesh_bounding_box.min.y, mesh_bounding_box.min.z, mesh_bounding_box.min.w);
    Vec128 v1 = vec128_set(mesh_bounding_box.min.x, mesh_bounding_box.min.y, mesh_bounding_box.max.z, mesh_bounding_box.min.w);
    Vec128 v2 = vec128_set(mesh_bounding_box.min.x, mesh_bounding_box.max.y, mesh_bounding_box.min.z, mesh_bounding_box.min.w);
    Vec128 v3 = vec128_set(mesh_bounding_box.min.x, mesh_bounding_box.max.y, mesh_bounding_box.max.z, mesh_bounding_box.min.w);
    Vec128 v4 = vec128_set(mesh_bounding_box.max.x, mesh_bounding_box.min.y, mesh_bounding_box.min.z, mesh_bounding_box.min.w);
    Vec128 v5 = vec128_set(mesh_bounding_box.max.x, mesh_bounding_box.min.y, mesh_bounding_box.max.z, mesh_bounding_box.min.w);
    Vec128 v6 = vec128_set(mesh_bounding_box.max.x, mesh_bounding_box.max.y, mesh_bounding_box.min.z, mesh_bounding_box.min.w);
    Vec128 v7 = vec128_set(mesh_bounding_box.max.x, mesh_bounding_box.max.y, mesh_bounding_box.max.z, mesh_bounding_box.min.w);

    v0 = vec128_add(vec128_mul(vec128_rotate_quaternion(v0, rotation), scale), offset);
    v1 = vec128_add(vec128_mul(vec128_rotate_quaternion(v1, rotation), scale), offset);
    v2 = vec128_add(vec128_mul(vec128_rotate_quaternion(v2, rotation), scale), offset);
    v3 = vec128_add(vec128_mul(vec128_rotate_quaternion(v3, rotation), scale), offset);
    v4 = vec128_add(vec128_mul(vec128_rotate_quaternion(v4, rotation), scale), offset);
    v5 = vec128_add(vec128_mul(vec128_rotate_quaternion(v5, rotation), scale), offset);
    v6 = vec128_add(vec128_mul(vec128_rotate_quaternion(v6, rotation), scale), offset);
    v7 = vec128_add(vec128_mul(vec128_rotate_quaternion(v7, rotation), scale), offset);

    min = vec128_min(min, v0);
    min = vec128_min(min, v1);
    min = vec128_min(min, v2);
    min = vec128_min(min, v3);
    min = vec128_min(min, v4);
    min = vec128_min(min, v5);
    min = vec128_min(min, v6);
    min = vec128_min(min, v7);

    max = vec128_max(max, v0);
    max = vec128_max(max, v1);
    max = vec128_max(max, v2);
    max = vec128_max(max, v3);
    max = vec128_max(max, v4);
    max = vec128_max(max, v5);
    max = vec128_max(max, v6);
    max = vec128_max(max, v7);
  }

  *min_bounds = min;
  *max_bounds = max;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_update(LightGridCache* cache, bool* requires_rebuild) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(requires_rebuild);

  Vec128 new_min;
  Vec128 new_max;
  __FAILURE_HANDLE(_light_grid_cache_get_scene_bounding_box(cache, &new_min, &new_max));

  cache->is_dirty = false;

  if (vec128_is_equal(new_min, cache->min_bounds) && vec128_is_equal(new_max, cache->max_bounds)) {
    *requires_rebuild = false;
  }
  else {
    cache->min_bounds = new_min;
    cache->max_bounds = new_max;
    *requires_rebuild = true;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_destroy(LightGridCache** cache) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(*cache);

  __FAILURE_HANDLE(array_destroy(&(*cache)->mesh_bounding_boxes));
  __FAILURE_HANDLE(array_destroy(&(*cache)->instances));

  __FAILURE_HANDLE(host_free(cache));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Build Implementation
////////////////////////////////////////////////////////////////////

static LuminaryResult _light_grid_free_data(LightGrid* light_grid) {
  __CHECK_NULL_ARGUMENT(light_grid);

  if (light_grid->cache_points_data) {
    __FAILURE_HANDLE(device_free(&light_grid->cache_points_data));
  }

  if (light_grid->cache_points_meta_data) {
    __FAILURE_HANDLE(device_free(&light_grid->cache_points_meta_data));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_generate_points(LightGrid* light_grid, Device* device) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(device);

  // TODO: The number of points should be determined by an upper bound on the amount of memory we want to use.
  // Say we want to use N bytes, create as many cache points such that we use N bytes in the worst case.
  // If a reasonable amount of bytes is left available after pruning, create more points.

  const LightGridCache* cache = (LightGridCache*) light_grid->cache;

  const uint32_t num_points                = 32;
  const uint32_t maximum_entries_per_point = 64;

  // Due to the prefix-sum, we need a multiple of 32 points.
  __DEBUG_ASSERT((num_points & ((1 << 5) - 1)) == 0);

  DEVICE MinHeapEntry* min_heap_buffer;
  __FAILURE_HANDLE(device_malloc(&min_heap_buffer, sizeof(MinHeapEntry) * maximum_entries_per_point * num_points));

  __FAILURE_HANDLE(device_malloc(&light_grid->cache_points_meta_data, sizeof(DeviceLightCachePointMeta) * num_points));

  DEVICE uint32_t* device_total_num_entries;
  __FAILURE_HANDLE(device_malloc(&device_total_num_entries, sizeof(uint32_t)));

  const vec3 bounds_min  = (vec3) {.x = cache->min_bounds.x, .y = cache->min_bounds.y, .z = cache->min_bounds.z};
  const vec3 bounds_max  = (vec3) {.x = cache->max_bounds.x, .y = cache->max_bounds.y, .z = cache->max_bounds.z};
  const vec3 bounds_span = (vec3) {.x = bounds_max.x - bounds_min.x, .y = bounds_max.y - bounds_min.y, .z = bounds_max.z - bounds_min.z};

  KernelArgsLightGridGenerate light_grid_generate_args;
  light_grid_generate_args.count                = num_points;
  light_grid_generate_args.bounds_min           = bounds_min;
  light_grid_generate_args.bounds_span          = bounds_span;
  light_grid_generate_args.allocated_entries    = maximum_entries_per_point;
  light_grid_generate_args.importance_threshold = 0.99f;
  light_grid_generate_args.min_heap_data        = DEVICE_PTR(min_heap_buffer);
  light_grid_generate_args.total_num_entries    = DEVICE_PTR(device_total_num_entries);
  light_grid_generate_args.dst_cache_point_meta = DEVICE_PTR(light_grid->cache_points_meta_data);

  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_LIGHT_GRID_GENERATE], &light_grid_generate_args, device->stream_main));

  uint32_t total_num_entries;
  __FAILURE_HANDLE(device_download(&total_num_entries, device_total_num_entries, 0, sizeof(uint32_t), device->stream_main));

  __FAILURE_HANDLE(device_malloc(&light_grid->cache_points_data, sizeof(DeviceLightCacheEntry) * total_num_entries));

  // TODO: Keep work buffers resident for fast rebuilds, I can split the work to keep the buffers small
  __FAILURE_HANDLE(device_free(&device_total_num_entries));
  __FAILURE_HANDLE(device_free(&min_heap_buffer));

  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}

////////////////////////////////////////////////////////////////////
// API implementations
////////////////////////////////////////////////////////////////////

LuminaryResult light_grid_create(LightGrid** light_grid) {
  __CHECK_NULL_ARGUMENT(light_grid);

  __FAILURE_HANDLE(host_malloc(light_grid, sizeof(LightGrid)));
  memset(*light_grid, 0, sizeof(LightGrid));

  __FAILURE_HANDLE(_light_grid_cache_create((LightGridCache**) &(*light_grid)->cache));

  return LUMINARY_SUCCESS;
}

LuminaryResult light_grid_update_cache_mesh(LightGrid* light_grid, const Mesh* mesh) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(mesh);

  __FAILURE_HANDLE(_light_grid_cache_update_mesh((LightGridCache*) light_grid->cache, mesh));

  return LUMINARY_SUCCESS;
}

LuminaryResult light_grid_update_cache_instance(LightGrid* light_grid, const MeshInstance* instance) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(instance);

  __FAILURE_HANDLE(_light_grid_cache_update_instance((LightGridCache*) light_grid->cache, instance));

  return LUMINARY_SUCCESS;
}

LuminaryResult light_grid_build(LightGrid* light_grid, Device* device) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(device);

  LightGridCache* cache = (LightGridCache*) light_grid->cache;

  // Only build if the cache is dirty.
  if (cache->is_dirty == false)
    return LUMINARY_SUCCESS;

  bool requires_rebuild;
  __FAILURE_HANDLE(_light_grid_cache_update(light_grid->cache, &requires_rebuild));

  if (requires_rebuild == false)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(_light_grid_free_data(light_grid));
  __FAILURE_HANDLE(_light_grid_generate_points(light_grid, device));

  light_grid->build_id++;

  return LUMINARY_SUCCESS;
}

LuminaryResult light_grid_destroy(LightGrid** light_grid) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(*light_grid);

  __FAILURE_HANDLE(_light_grid_cache_destroy((LightGridCache**) &(*light_grid)->cache));

  __FAILURE_HANDLE(host_free(light_grid));

  return LUMINARY_SUCCESS;
}
