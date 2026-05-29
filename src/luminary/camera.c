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
  camera->aperture_shape               = LUMINARY_APERTURE_BLADED;
  camera->aperture_blade_count         = 7;
  camera->exposure                     = 0.0f;
  camera->shutter_speed                = 200.0f;
  camera->bloom_blend                  = 0.01f;
  camera->dithering                    = 1;
  camera->tonemap                      = LUMINARY_TONEMAP_AGX;
  camera->use_local_error_minimization = false;
  camera->purkinje                     = false;
  camera->purkinje_kappa1              = 0.2f;
  camera->purkinje_kappa2              = 0.29f;
  camera->lens_template                = LUMINARY_LENS_TEMPLATE_PHYSICAL_G;
  camera->thin_lens.aperture_size      = 0.0f;
  camera->thin_lens.object_distance    = 1.0f;
  camera->thin_lens.fov                = 1.0f;
  camera->lens.f_stop                  = 2.0f;
  camera->lens.sensor_distance         = 40.0f;
  camera->lens.object_distance         = 10.0f;
  camera->lens.scale                   = 1.0f;
  camera->lens.use_auto_focus          = true;
  camera->lens.use_spectral_rendering  = false;
  camera->lens.allow_reflections       = false;
  camera->lens.enable_diffraction      = false;

  camera->sensor = (LuminaryCameraSensor) {
    .diagonal_size                    = 43.3f,
    .aspect_ratio                     = 16.0f / 9.0f,
    .use_aspect_ratio_from_resolution = true,
    .iso                              = 200.0f,
    .film_grain_strength              = 0.0f,
    .response_model                   = LUMINARY_SENSOR_RESPONSE_IDEAL,
    .film_thickness                   = 1.0f,
    .microlens_acceptance_angle       = PI * (25.0f / 180.0f),
  };

  camera->tonemap_params = (LuminaryTonemapParams) {
    .highlights                = 0.0f,
    .shadows                   = 0.0f,
    .saturation                = 0.0f,
    .dynamic_range             = 100.0f,
    .white_balance_red_cyan    = 0.0f,
    .white_balance_blue_yellow = 0.0f,
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
  __CAMERA_CHECK_DIRTY(lens_template, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
  __CAMERA_CHECK_DIRTY(aperture_shape, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  // TODO: This condition will be whether motion blur is enabled or not.
  if (false) {
    __CAMERA_CHECK_DIRTY(shutter_speed, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }
  else {
    __CAMERA_CHECK_DIRTY(shutter_speed, SCENE_DIRTY_FLAG_OUTPUT);
  }

  if (input->aperture_shape != LUMINARY_APERTURE_ROUND) {
    __CAMERA_CHECK_DIRTY(aperture_blade_count, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  if (input->lens_template == LUMINARY_LENS_TEMPLATE_THIN_LENS) {
    __CAMERA_CHECK_DIRTY(thin_lens.aperture_size, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(thin_lens.object_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(thin_lens.fov, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }
  else {
    __CAMERA_CHECK_DIRTY(lens.f_stop, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
    __CAMERA_CHECK_DIRTY(lens.sensor_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
    __CAMERA_CHECK_DIRTY(lens.use_auto_focus, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
    __CAMERA_CHECK_DIRTY(lens.object_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
    __CAMERA_CHECK_DIRTY(lens.scale, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT | SCENE_DIRTY_FLAG_CAMERA_TEMPLATE);
    __CAMERA_CHECK_DIRTY(lens.enable_diffraction, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(lens.use_spectral_rendering, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(lens.allow_reflections, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __CAMERA_CHECK_DIRTY(sensor.diagonal_size, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.aspect_ratio, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.use_aspect_ratio_from_resolution, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.iso, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(sensor.film_grain_strength, SCENE_DIRTY_FLAG_OUTPUT);

  __CAMERA_CHECK_DIRTY(sensor.response_model, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  if (input->sensor.response_model == LUMINARY_SENSOR_RESPONSE_FILM) {
    __CAMERA_CHECK_DIRTY(sensor.film_thickness, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }
  else if (input->sensor.response_model == LUMINARY_SENSOR_RESPONSE_DIGITAL) {
    __CAMERA_CHECK_DIRTY(sensor.microlens_acceptance_angle, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __CAMERA_CHECK_DIRTY(use_local_error_minimization, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(exposure, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(bloom_blend, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(dithering, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa1, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa2, SCENE_DIRTY_FLAG_OUTPUT);

  __CAMERA_CHECK_DIRTY(tonemap_params.highlights, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap_params.shadows, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap_params.saturation, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap_params.dynamic_range, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap_params.white_balance_red_cyan, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap_params.white_balance_blue_yellow, SCENE_DIRTY_FLAG_OUTPUT);

  return LUMINARY_SUCCESS;
}
