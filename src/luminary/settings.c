#include "settings.h"

#include "internal_error.h"

LuminaryResult settings_get_default(RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(settings);

  settings->width                        = 1920;
  settings->height                       = 1080;
  settings->max_ray_depth                = 4;
  settings->use_denoiser                 = false;
  settings->bridge_max_num_vertices      = 1;
  settings->bridge_num_ris_samples       = 8;
  settings->light_initial_reservoir_size = 16;
  settings->light_num_rays               = 1;
  settings->use_opacity_micromaps        = false;
  settings->use_displacement_micromaps   = false;
  settings->enable_optix_validation      = false;
  settings->use_luminary_bvh             = false;
  settings->undersampling                = 3;
  settings->shading_mode                 = LUMINARY_SHADING_MODE_DEFAULT;

  return LUMINARY_SUCCESS;
}

#define __SETTINGS_DIRTY(var)   \
  {                             \
    if (new->var != old->var) { \
      *dirty = true;            \
      return LUMINARY_SUCCESS;  \
    }                           \
  }

LuminaryResult settings_check_for_dirty(const RendererSettings* new, const RendererSettings* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(new);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __SETTINGS_DIRTY(width);
  __SETTINGS_DIRTY(height);
  __SETTINGS_DIRTY(use_denoiser);
  __SETTINGS_DIRTY(bridge_max_num_vertices);
  __SETTINGS_DIRTY(bridge_num_ris_samples);
  __SETTINGS_DIRTY(light_initial_reservoir_size);
  __SETTINGS_DIRTY(light_num_rays);
  __SETTINGS_DIRTY(use_opacity_micromaps);
  __SETTINGS_DIRTY(use_displacement_micromaps);
  __SETTINGS_DIRTY(enable_optix_validation);
  __SETTINGS_DIRTY(use_luminary_bvh);
  __SETTINGS_DIRTY(undersampling);

  if (new->shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
    __SETTINGS_DIRTY(max_ray_depth);
  }

  return LUMINARY_SUCCESS;
}
