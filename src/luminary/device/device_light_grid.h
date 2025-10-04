#ifndef LUMINARY_DEVICE_LIGHT_GRID_H
#define LUMINARY_DEVICE_LIGHT_GRID_H

#include "device_light.h"
#include "device_utils.h"

struct LightGrid {
  uint32_t build_id;
  DEVICE DeviceLightCacheEntry* cache_points_data;
  DEVICE DeviceLightCachePointMeta* cache_points_meta_data;
  void* cache;
} typedef LightGrid;

LuminaryResult light_grid_create(LightGrid** light_grid);
LuminaryResult light_grid_update_cache_mesh(LightGrid* light_grid, const Mesh* mesh);
LuminaryResult light_grid_update_cache_instance(LightGrid* light_grid, const MeshInstance* instance);
DEVICE_CTX_FUNC LuminaryResult light_grid_build(LightGrid* light_grid, Device* device);
LuminaryResult light_grid_destroy(LightGrid** light_grid);

#endif /* LUMINARY_DEVICE_LIGHT_GRID_H */
