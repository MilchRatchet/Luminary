#include "sky.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult sky_get_default(Sky* sky) {
  __CHECK_NULL_ARGUMENT(sky);

  sky->geometry_offset.x      = 0.0f;
  sky->geometry_offset.y      = 0.1f;
  sky->geometry_offset.z      = 0.0f;
  sky->altitude               = 0.5f;
  sky->azimuth                = 3.141f;
  sky->moon_altitude          = -0.5f;
  sky->moon_azimuth           = 0.0f;
  sky->moon_tex_offset        = 0.0f;
  sky->sun_strength           = 1.0f;
  sky->base_density           = 1.0f;
  sky->rayleigh_density       = 1.0f;
  sky->mie_density            = 1.0f;
  sky->ozone_density          = 1.0f;
  sky->ground_visibility      = 60.0f;
  sky->mie_diameter           = 2.0f;
  sky->ozone_layer_thickness  = 15.0f;
  sky->rayleigh_falloff       = 8.0f;
  sky->mie_falloff            = 1.7f;
  sky->multiscattering_factor = 1.0f;
  sky->steps                  = 40;
  sky->ozone_absorption       = true;
  sky->aerial_perspective     = false;
  sky->hdri_dim               = 2048;
  sky->hdri_samples           = 32;
  sky->stars_seed             = 0;
  sky->stars_count            = 10000;
  sky->stars_intensity        = 1.0f;
  sky->constant_color.r       = 1.0f;
  sky->constant_color.g       = 1.0f;
  sky->constant_color.b       = 1.0f;
  sky->mode                   = LUMINARY_SKY_MODE_DEFAULT;

  return LUMINARY_SUCCESS;
}

#define SKY_ALL_DIRTY_FLAGS \
  ((uint32_t) (SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_HDRI | SCENE_DIRTY_FLAG_PASSIVE))

#define __SKY_CHECK_DIRTY(var, flags)                                  \
  {                                                                    \
    if (input->var != old->var) {                                      \
      *dirty_flags |= flags;                                           \
      if ((*dirty_flags & SKY_ALL_DIRTY_FLAGS) == SKY_ALL_DIRTY_FLAGS) \
        return LUMINARY_SUCCESS;                                       \
    }                                                                  \
  }

LuminaryResult sky_check_for_dirty(const Sky* input, const Sky* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __SKY_CHECK_DIRTY(mode, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_HDRI);

  switch (input->mode) {
    case LUMINARY_SKY_MODE_DEFAULT:
      __SKY_CHECK_DIRTY(geometry_offset.x, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(geometry_offset.y, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(geometry_offset.z, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(altitude, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(azimuth, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(moon_altitude, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(moon_azimuth, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(moon_tex_offset, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(sun_strength, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(base_density, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(ozone_absorption, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(steps, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(stars_intensity, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(rayleigh_density, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(mie_density, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(ozone_density, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(rayleigh_falloff, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(mie_falloff, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(mie_diameter, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(ground_visibility, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(ozone_layer_thickness, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(multiscattering_factor, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(rayleigh_density, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(aerial_perspective, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(stars_count, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(stars_seed, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(stars_intensity, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      break;
    case LUMINARY_SKY_MODE_HDRI:
      __SKY_CHECK_DIRTY(altitude, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(azimuth, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(sun_strength, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(aerial_perspective, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(hdri_dim, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_PASSIVE);
      __SKY_CHECK_DIRTY(hdri_samples, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_PASSIVE);

      break;
    case LUMINARY_SKY_MODE_CONSTANT_COLOR:
      __SKY_CHECK_DIRTY(constant_color.r, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(constant_color.g, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      __SKY_CHECK_DIRTY(constant_color.b, SCENE_DIRTY_FLAG_SKY | SCENE_DIRTY_FLAG_INTEGRATION);
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Invalid sky mode.");
  }

  return LUMINARY_SUCCESS;
}
