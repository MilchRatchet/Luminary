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
  sky->stars_intensity        = 1.0f;
  sky->settings_stars_count   = 10000;
  sky->constant_color.r       = 1.0f;
  sky->constant_color.g       = 1.0f;
  sky->constant_color.b       = 1.0f;
  sky->ambient_sampling       = true;
  sky->mode                   = LUMINARY_SKY_MODE_DEFAULT;

  return LUMINARY_SUCCESS;
}
