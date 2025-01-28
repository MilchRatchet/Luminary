#include "sky.h"

#include "internal_error.h"

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
  sky->lut_initialized        = 0;
  sky->hdri_initialized       = 0;
  sky->hdri_dim               = 0;
  sky->settings_hdri_dim      = 2048;
  sky->hdri_samples           = 50;
  sky->hdri_origin.x          = 0.0f;
  sky->hdri_origin.y          = 1.0f;
  sky->hdri_origin.z          = 0.0f;
  sky->hdri_mip_bias          = 0.0f;
  sky->stars_seed             = 0;
  sky->stars_count            = 10000;
  sky->stars_intensity        = 1.0f;
  sky->constant_color.r       = 1.0f;
  sky->constant_color.g       = 1.0f;
  sky->constant_color.b       = 1.0f;
  sky->ambient_sampling       = true;
  sky->mode                   = LUMINARY_SKY_MODE_DEFAULT;

  return LUMINARY_SUCCESS;
}

#define __SKY_DIRTY(var)          \
  {                               \
    if (input->var != old->var) { \
      *dirty = true;              \
      return LUMINARY_SUCCESS;    \
    }                             \
  }

LuminaryResult sky_check_for_dirty(const Sky* input, const Sky* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __SKY_DIRTY(mode);

  __SKY_DIRTY(ambient_sampling);

  switch (input->mode) {
    case LUMINARY_SKY_MODE_DEFAULT:
      __SKY_DIRTY(geometry_offset.x);
      __SKY_DIRTY(geometry_offset.y);
      __SKY_DIRTY(geometry_offset.z);
      __SKY_DIRTY(altitude);
      __SKY_DIRTY(azimuth);
      __SKY_DIRTY(moon_altitude);
      __SKY_DIRTY(moon_azimuth);
      __SKY_DIRTY(moon_tex_offset);
      __SKY_DIRTY(sun_strength);
      __SKY_DIRTY(base_density);
      __SKY_DIRTY(ozone_absorption);
      __SKY_DIRTY(steps);
      __SKY_DIRTY(stars_intensity);
      __SKY_DIRTY(rayleigh_density);
      __SKY_DIRTY(mie_density);
      __SKY_DIRTY(ozone_density);
      __SKY_DIRTY(rayleigh_falloff);
      __SKY_DIRTY(mie_falloff);
      __SKY_DIRTY(mie_diameter);
      __SKY_DIRTY(ground_visibility);
      __SKY_DIRTY(ozone_layer_thickness);
      __SKY_DIRTY(multiscattering_factor);
      __SKY_DIRTY(rayleigh_density);
      __SKY_DIRTY(aerial_perspective);
      __SKY_DIRTY(stars_count);
      __SKY_DIRTY(stars_seed);
      __SKY_DIRTY(stars_intensity);
      break;
    case LUMINARY_SKY_MODE_HDRI:
      __SKY_DIRTY(altitude);
      __SKY_DIRTY(azimuth);
      __SKY_DIRTY(sun_strength);
      __SKY_DIRTY(hdri_mip_bias);
      __SKY_DIRTY(aerial_perspective);
      break;
    case LUMINARY_SKY_MODE_CONSTANT_COLOR:
      __SKY_DIRTY(constant_color.r);
      __SKY_DIRTY(constant_color.g);
      __SKY_DIRTY(constant_color.b);
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Invalid sky mode.");
  }

  return LUMINARY_SUCCESS;
}
