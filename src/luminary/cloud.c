#include "cloud.h"

#include "internal_error.h"
#include "scene.h"

LuminaryResult cloud_get_default(Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(cloud);

  cloud->active                = false;
  cloud->initialized           = false;
  cloud->steps                 = 96;
  cloud->shadow_steps          = 8;
  cloud->atmosphere_scattering = true;
  cloud->seed                  = CLOUD_DEFAULT_SEED;
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

#define __CLOUD_CHECK_DIRTY(var)                                            \
  {                                                                         \
    if (input->var != old->var) {                                           \
      *dirty_flags = SCENE_DIRTY_FLAG_CLOUD | SCENE_DIRTY_FLAG_INTEGRATION; \
      return LUMINARY_SUCCESS;                                              \
    }                                                                       \
  }

static LuminaryResult _cloud_layer_check_for_dirty(const CloudLayer* input, const CloudLayer* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __CLOUD_CHECK_DIRTY(active);

  if (input->active) {
    __CLOUD_CHECK_DIRTY(height_max);
    __CLOUD_CHECK_DIRTY(height_min);
    __CLOUD_CHECK_DIRTY(coverage);
    __CLOUD_CHECK_DIRTY(coverage_min);
    __CLOUD_CHECK_DIRTY(type);
    __CLOUD_CHECK_DIRTY(type_min);
    __CLOUD_CHECK_DIRTY(wind_speed);
    __CLOUD_CHECK_DIRTY(wind_angle);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult cloud_check_for_dirty(const Cloud* input, const Cloud* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __CLOUD_CHECK_DIRTY(active);

  if (input->active) {
    __CLOUD_CHECK_DIRTY(atmosphere_scattering);
    __CLOUD_CHECK_DIRTY(offset_x);
    __CLOUD_CHECK_DIRTY(offset_z);
    __CLOUD_CHECK_DIRTY(density);
    __CLOUD_CHECK_DIRTY(seed);
    __CLOUD_CHECK_DIRTY(droplet_diameter);
    __CLOUD_CHECK_DIRTY(steps);
    __CLOUD_CHECK_DIRTY(shadow_steps);
    __CLOUD_CHECK_DIRTY(noise_shape_scale);
    __CLOUD_CHECK_DIRTY(noise_detail_scale);
    __CLOUD_CHECK_DIRTY(noise_weather_scale);
    __CLOUD_CHECK_DIRTY(mipmap_bias);
    __CLOUD_CHECK_DIRTY(octaves);

    __FAILURE_HANDLE(_cloud_layer_check_for_dirty(&input->low, &old->low, dirty_flags));
    if (*dirty_flags & SCENE_DIRTY_FLAG_CLOUD) {
      return LUMINARY_SUCCESS;
    }

    __FAILURE_HANDLE(_cloud_layer_check_for_dirty(&input->mid, &old->mid, dirty_flags));
    if (*dirty_flags & SCENE_DIRTY_FLAG_CLOUD) {
      return LUMINARY_SUCCESS;
    }

    __FAILURE_HANDLE(_cloud_layer_check_for_dirty(&input->top, &old->top, dirty_flags));
    if (*dirty_flags & SCENE_DIRTY_FLAG_CLOUD) {
      return LUMINARY_SUCCESS;
    }
  }

  return LUMINARY_SUCCESS;
}
