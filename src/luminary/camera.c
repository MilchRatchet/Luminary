#include "camera.h"

#include "internal_error.h"
#include "lens_library.h"
#include "scene.h"
#include "utils.h"

LuminaryResult camera_get_default(Camera* camera) {
  __CHECK_NULL_ARGUMENT(camera);

  camera->pos.x                        = 0.0f;
  camera->pos.y                        = 0.0f;
  camera->pos.z                        = 0.0f;
  camera->rotation.x                   = 0.0f;
  camera->rotation.y                   = 0.0f;
  camera->rotation.z                   = 0.0f;
  camera->aperture_shape               = LUMINARY_APERTURE_ROUND;
  camera->aperture_blade_count         = 7;
  camera->exposure                     = 0.0f;
  camera->bloom_blend                  = 0.01f;
  camera->dithering                    = 1;
  camera->tonemap                      = LUMINARY_TONEMAP_AGX;
  camera->use_local_error_minimization = false;
  camera->agx_custom_slope             = 1.0f;
  camera->agx_custom_power             = 1.0f;
  camera->agx_custom_saturation        = 1.0f;
  camera->purkinje                     = 1;
  camera->purkinje_kappa1              = 0.2f;
  camera->purkinje_kappa2              = 0.29f;
  camera->use_color_correction         = 0;
  camera->color_correction.r           = 0.0f;
  camera->color_correction.g           = 0.0f;
  camera->color_correction.b           = 0.0f;
  camera->scale                        = 1.0f;
  camera->use_spectral_rendering       = false;
  camera->allow_reflections            = false;
  camera->lens_template                = LUMINARY_LENS_TEMPLATE_THIN_LENS;
  camera->lens.aperture_stop           = 1.0f;
  camera->lens.sensor_diagonal_size    = 1.0f;
  camera->lens.sensor_distance         = 1.0f;
  camera->lens.use_auto_focus          = false;
  camera->lens.object_distance         = 1.0f;

  camera->sensor = (LuminaryCameraSensor) {
    .aspect_ratio                     = 16.0f / 9.0f,
    .use_aspect_ratio_from_resolution = true,
    .film_grain_strength              = 0.0f,
    .film_grain_coarseness            = 0.0f,
    .film_grains_per_pixel            = 256,
  };

  return LUMINARY_SUCCESS;
}

#define CAMERA_ALL_DIRTY_FLAGS \
  ((uint32_t) (SCENE_DIRTY_FLAG_CAMERA | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE))

#define __CAMERA_CHECK_DIRTY(var, flags)                                     \
  {                                                                          \
    if (input->var != old->var) {                                            \
      *dirty_flags |= flags | SCENE_DIRTY_FLAG_CAMERA;                       \
      if ((*dirty_flags & CAMERA_ALL_DIRTY_FLAGS) == CAMERA_ALL_DIRTY_FLAGS) \
        return LUMINARY_SUCCESS;                                             \
    }                                                                        \
  }

LuminaryResult camera_check_for_dirty(const Camera* input, const Camera* old, uint32_t* dirty_flags) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty_flags);

  __CAMERA_CHECK_DIRTY(pos.x, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(pos.y, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(pos.z, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(rotation.x, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(rotation.y, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(rotation.z, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(scale, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
  __CAMERA_CHECK_DIRTY(lens_template, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);

  if (input->lens_template != LUMINARY_LENS_TEMPLATE_THIN_LENS) {
    __CAMERA_CHECK_DIRTY(use_spectral_rendering, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(allow_reflections, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __CAMERA_CHECK_DIRTY(aperture_shape, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  if (input->aperture_shape != LUMINARY_APERTURE_ROUND) {
    __CAMERA_CHECK_DIRTY(aperture_blade_count, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __CAMERA_CHECK_DIRTY(lens.aperture_stop, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
  __CAMERA_CHECK_DIRTY(lens.sensor_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
  __CAMERA_CHECK_DIRTY(lens.sensor_diagonal_size, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(lens.use_auto_focus, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
  __CAMERA_CHECK_DIRTY(lens.object_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);

  __CAMERA_CHECK_DIRTY(sensor.aspect_ratio, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.use_aspect_ratio_from_resolution, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.film_grain_strength, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.film_grain_coarseness, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.film_grains_per_pixel, SCENE_DIRTY_FLAG_OUTPUT);

  __CAMERA_CHECK_DIRTY(use_local_error_minimization, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(exposure, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(bloom_blend, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(dithering, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa1, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa2, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(use_color_correction, SCENE_DIRTY_FLAG_OUTPUT);

  if (input->tonemap == LUMINARY_TONEMAP_AGX_CUSTOM) {
    __CAMERA_CHECK_DIRTY(agx_custom_slope, SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(agx_custom_power, SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(agx_custom_saturation, SCENE_DIRTY_FLAG_OUTPUT);
  }

  if (input->use_color_correction) {
    __CAMERA_CHECK_DIRTY(color_correction.r, SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(color_correction.g, SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(color_correction.b, SCENE_DIRTY_FLAG_OUTPUT);
  }

  return LUMINARY_SUCCESS;
}
