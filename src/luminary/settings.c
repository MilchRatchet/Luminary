#include "settings.h"

#include "internal_error.h"

LuminaryResult settings_get_default(RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(settings);

  settings->width                      = 1920;
  settings->height                     = 1080;
  settings->max_ray_depth              = 4;
  settings->use_denoiser               = false;
  settings->bridge_max_num_vertices    = 1;
  settings->bridge_num_ris_samples     = 8;
  settings->light_num_ris_samples      = 16;
  settings->light_num_rays             = 1;
  settings->use_opacity_micromaps      = false;
  settings->use_displacement_micromaps = false;
  settings->use_luminary_bvh           = false;
  settings->undersampling              = 3;
  settings->shading_mode               = LUMINARY_SHADING_MODE_DEFAULT;
  settings->max_sample_count           = 0xFFFFFFFF;

  return LUMINARY_SUCCESS;
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

LuminaryResult settings_check_for_dirty(
  const RendererSettings* input, const RendererSettings* old, bool* integration_dirty, bool* buffers_dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(integration_dirty);
  __CHECK_NULL_ARGUMENT(buffers_dirty);

  *integration_dirty = false;
  *buffers_dirty     = false;

  __SETTINGS_STANDARD_DIRTY(width);
  __SETTINGS_STANDARD_DIRTY(height);
  __SETTINGS_STANDARD_DIRTY(use_denoiser);
  __SETTINGS_INTEGRATION_DIRTY(bridge_max_num_vertices);
  __SETTINGS_INTEGRATION_DIRTY(bridge_num_ris_samples);
  __SETTINGS_INTEGRATION_DIRTY(light_num_ris_samples);
  __SETTINGS_INTEGRATION_DIRTY(light_num_rays);
  __SETTINGS_INTEGRATION_DIRTY(use_opacity_micromaps);
  __SETTINGS_INTEGRATION_DIRTY(use_displacement_micromaps);
  __SETTINGS_INTEGRATION_DIRTY(use_luminary_bvh);
  __SETTINGS_INTEGRATION_DIRTY(undersampling);

  if (input->shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
    __SETTINGS_INTEGRATION_DIRTY(max_ray_depth);
  }

  return LUMINARY_SUCCESS;
}
