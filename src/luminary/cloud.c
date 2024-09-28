#include "cloud.h"

#include "internal_error.h"

LuminaryResult cloud_get_default(Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(cloud);

  cloud->active                = false;
  cloud->initialized           = false;
  cloud->steps                 = 96;
  cloud->shadow_steps          = 8;
  cloud->atmosphere_scattering = true;
  cloud->seed                  = 1;
  cloud->offset_x              = 0.0f;
  cloud->offset_z              = 0.0f;
  cloud->noise_shape_scale     = 1.0f;
  cloud->noise_detail_scale    = 1.0f;
  cloud->noise_weather_scale   = 1.0f;
  cloud->octaves               = 9;
  cloud->droplet_diameter      = 25.0f;
  cloud->density               = 1.0f;
  cloud->mipmap_bias           = 0.0f;
  cloud->low.active            = true;
  cloud->low.height_max        = 5.0f;
  cloud->low.height_min        = 1.5f;
  cloud->low.coverage          = 1.0f;
  cloud->low.coverage_min      = 0.0f;
  cloud->low.type              = 1.0f;
  cloud->low.type_min          = 0.0f;
  cloud->low.wind_speed        = 2.5f;
  cloud->low.wind_angle        = 0.0f;
  cloud->mid.active            = true;
  cloud->mid.height_max        = 6.0f;
  cloud->mid.height_min        = 5.5f;
  cloud->mid.coverage          = 1.0f;
  cloud->mid.coverage_min      = 0.0f;
  cloud->mid.type              = 1.0f;
  cloud->mid.type_min          = 0.0f;
  cloud->mid.wind_speed        = 2.5f;
  cloud->mid.wind_angle        = 0.0f;
  cloud->top.active            = true;
  cloud->top.height_max        = 8.0f;
  cloud->top.height_min        = 7.95f;
  cloud->top.coverage          = 1.0f;
  cloud->top.coverage_min      = 0.0f;
  cloud->top.type              = 1.0f;
  cloud->top.type_min          = 0.0f;
  cloud->top.wind_speed        = 1.0f;
  cloud->top.wind_angle        = 0.0f;

  return LUMINARY_SUCCESS;
}
