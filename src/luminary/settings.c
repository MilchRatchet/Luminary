#include "settings.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult settings_get_default(RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(settings);

  settings->width                               = 2560;
  settings->height                              = 1440;
  settings->max_ray_depth                       = 4;
  settings->bridge_max_num_vertices             = 15;
  settings->undersampling                       = 2;
  settings->supersampling                       = 1;
  settings->enable_adaptive_sampling            = true;
  settings->adaptive_sampling_max_sampling_rate = 256;
  settings->adaptive_sampling_avg_sampling_rate = 2;
  settings->adaptive_sampling_update_interval   = 64;
  settings->adaptive_sampling_exposure_aware    = true;
  settings->adaptive_sampling_output_mode       = LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_BEAUTY;
  settings->shading_mode                        = LUMINARY_SHADING_MODE_DEFAULT;
  settings->region_x                            = 0.0f;
  settings->region_y                            = 0.0f;
  settings->region_width                        = 1.0f;
  settings->region_height                       = 1.0f;

  return LUMINARY_SUCCESS;
}

#define SETTINGS_ALL_DIRTY_FLAGS \
  ((uint32_t) (SCENE_DIRTY_FLAG_SETTINGS | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS | SCENE_DIRTY_FLAG_OUTPUT))

#define __SETTINGS_CHECK_DIRTY(var, flags)                                       \
  {                                                                              \
    if (input->var != old->var) {                                                \
      *dirty_flags |= flags | SCENE_DIRTY_FLAG_SETTINGS;                         \
      if ((*dirty_flags & SETTINGS_ALL_DIRTY_FLAGS) == SETTINGS_ALL_DIRTY_FLAGS) \
        return LUMINARY_SUCCESS;                                                 \
    }                                                                            \
  }

LuminaryResult settings_check_for_dirty(const RendererSettings* input, const RendererSettings* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __SETTINGS_CHECK_DIRTY(width, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(height, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(supersampling, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(bridge_max_num_vertices, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(undersampling, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(shading_mode, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(enable_adaptive_sampling, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_BUFFERS | SCENE_DIRTY_FLAG_OUTPUT);

  if (input->enable_adaptive_sampling) {
    __SETTINGS_CHECK_DIRTY(adaptive_sampling_max_sampling_rate, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __SETTINGS_CHECK_DIRTY(adaptive_sampling_avg_sampling_rate, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __SETTINGS_CHECK_DIRTY(adaptive_sampling_update_interval, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __SETTINGS_CHECK_DIRTY(adaptive_sampling_exposure_aware, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __SETTINGS_CHECK_DIRTY(adaptive_sampling_output_mode, SCENE_DIRTY_FLAG_OUTPUT);
  }

  if (input->shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
    __SETTINGS_CHECK_DIRTY(max_ray_depth, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __SETTINGS_CHECK_DIRTY(region_x, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(region_y, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(region_width, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __SETTINGS_CHECK_DIRTY(region_height, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  return LUMINARY_SUCCESS;
}
