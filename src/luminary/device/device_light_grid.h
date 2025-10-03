#ifndef LUMINARY_DEVICE_LIGHT_GRID_H
#define LUMINARY_DEVICE_LIGHT_GRID_H

#include "device_light.h"
#include "device_utils.h"

struct LightGrid {
  void* cache;
} typedef LightGrid;

LuminaryResult light_grid_create(LightGrid** light_grid);
LuminaryResult light_grid_update_cache_instance(LightGrid* light_grid, const MeshInstance* instance);
LuminaryResult light_grid_build(LightGrid* light_grid, Device* device, LightTree* light_tree, const Mesh** meshes);
LuminaryResult light_grid_destroy(LightGrid** light_grid);

#endif /* LUMINARY_DEVICE_LIGHT_GRID_H */
