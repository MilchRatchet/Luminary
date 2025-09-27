#include "device_light_grid.h"

#include "internal_error.h"

////////////////////////////////////////////////////////////////////
// Light Grid Cache Implementation
////////////////////////////////////////////////////////////////////

struct LightGridCache {
} typedef LightGridCache;

static LuminaryResult _light_grid_cache_create(LightGridCache** cache) {
  __CHECK_NULL_ARGUMENT(cache);

  __FAILURE_HANDLE(host_malloc(cache, sizeof(LightGridCache)));
  memset(*cache, 0, sizeof(LightGridCache));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _light_grid_cache_destroy(LightGridCache** cache) {
  __CHECK_NULL_ARGUMENT(cache);
  __CHECK_NULL_ARGUMENT(*cache);

  __FAILURE_HANDLE(host_free(cache));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Build Implementation
////////////////////////////////////////////////////////////////////

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

LuminaryResult light_grid_add_mesh(LightGrid* light_grid, const Mesh* mesh);
LuminaryResult light_grid_add_instance(LightGrid* light_grid, const MeshInstance* instance);

LuminaryResult light_grid_build(LightGrid* light_grid, LightTree* light_tree) {
  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}

LuminaryResult light_grid_destroy(LightGrid** light_grid) {
  __CHECK_NULL_ARGUMENT(light_grid);
  __CHECK_NULL_ARGUMENT(*light_grid);

  __FAILURE_HANDLE(_light_grid_cache_destroy((LightGridCache**) &(*light_grid)->cache));

  __FAILURE_HANDLE(host_free(light_grid));

  return LUMINARY_SUCCESS;
}
