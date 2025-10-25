#include "settings.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult settings_get_default(RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(settings);

  settings->width                   = 2560;
  settings->height                  = 1440;
  settings->max_ray_depth           = 4;
  settings->bridge_max_num_vertices = 15;
  settings->undersampling           = 2;
  settings->supersampling           = 1;
  settings->shading_mode            = LUMINARY_SHADING_MODE_DEFAULT;
  settings->max_sample_count        = 0xFFFFFFFF;
  settings->region_x                = 0.0f;
  settings->region_y                = 0.0f;
  settings->region_width            = 1.0f;
  settings->region_height           = 1.0f;

  return LUMINARY_SUCCESS;
}

#define SETTINGS_ALL_DIRTY_FLAGS ((uint32_t) (SCENE_DIRTY_FLAG_SETTINGS | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS))

#define __SETTINGS_CHECK_DIRTY(var, flags)                                       \
  {                                                                              \
    if (input->var != old->var) {                                                \
      *dirty_flags |= flags | SCENE_DIRTY_FLAG_SETTINGS;                         \
      if ((*dirty_flags & SETTINGS_ALL_DIRTY_FLAGS) == SETTINGS_ALL_DIRTY_FLAGS) \
        return LUMINARY_SUCCESS;                                                 \
    }                                                                            \
  }

#define __SETTINGS_STANDARD_DIRTY(var) \
  {                                    \
    if (input->var != old->var) {      \
      *integration_dirty = true;       \
      *buffers_dirty     = true;       \
      return LUMINARY_SUCCESS;         \
    }                                  \
  }

#define __SETTINGS_INTEGRATION_DIRTY(var) \
  {                                       \
    if (input->var != old->var) {         \
      *integration_dirty = true;          \
    }                                     \
  }

LuminaryResult settings_check_for_dirty(const RendererSettings* input, const RendererSettings* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __SETTINGS_CHECK_DIRTY(width, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS);
  __SETTINGS_CHECK_DIRTY(height, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS);
  __SETTINGS_CHECK_DIRTY(supersampling, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS);
  __SETTINGS_CHECK_DIRTY(bridge_max_num_vertices, SCENE_DIRTY_FLAG_INTEGRATION);
  __SETTINGS_CHECK_DIRTY(undersampling, SCENE_DIRTY_FLAG_INTEGRATION);
  __SETTINGS_CHECK_DIRTY(shading_mode, SCENE_DIRTY_FLAG_INTEGRATION);

  if (input->shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
    __SETTINGS_CHECK_DIRTY(max_ray_depth, SCENE_DIRTY_FLAG_INTEGRATION);
  }

  __SETTINGS_CHECK_DIRTY(region_x, SCENE_DIRTY_FLAG_INTEGRATION);
  __SETTINGS_CHECK_DIRTY(region_y, SCENE_DIRTY_FLAG_INTEGRATION);
  __SETTINGS_CHECK_DIRTY(region_width, SCENE_DIRTY_FLAG_INTEGRATION);
  __SETTINGS_CHECK_DIRTY(region_height, SCENE_DIRTY_FLAG_INTEGRATION);

  return LUMINARY_SUCCESS;
}
