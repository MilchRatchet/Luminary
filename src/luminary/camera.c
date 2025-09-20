#include "camera.h"

#include "internal_error.h"
#include "utils.h"

LuminaryResult camera_get_default(Camera* camera) {
  __CHECK_NULL_ARGUMENT(camera);

  camera->pos.x                      = 0.0f;
  camera->pos.y                      = 0.0f;
  camera->pos.z                      = 0.0f;
  camera->rotation.x                 = 0.0f;
  camera->rotation.y                 = 0.0f;
  camera->rotation.z                 = 0.0f;
  camera->aperture_shape             = LUMINARY_APERTURE_ROUND;
  camera->aperture_blade_count       = 7;
  camera->exposure                   = 0.0f;
  camera->bloom_blend                = 0.01f;
  camera->lens_flare                 = 0;
  camera->lens_flare_threshold       = 1.0f;
  camera->dithering                  = 1;
  camera->tonemap                    = LUMINARY_TONEMAP_AGX;
  camera->agx_custom_slope           = 1.0f;
  camera->agx_custom_power           = 1.0f;
  camera->agx_custom_saturation      = 1.0f;
  camera->filter                     = LUMINARY_FILTER_NONE;
  camera->wasd_speed                 = 1.0f;
  camera->mouse_speed                = 1.0f;
  camera->smooth_movement            = 0;
  camera->smoothing_factor           = 0.1f;
  camera->purkinje                   = 1;
  camera->purkinje_kappa1            = 0.2f;
  camera->purkinje_kappa2            = 0.29f;
  camera->russian_roulette_threshold = 0.1f;
  camera->use_color_correction       = 0;
  camera->color_correction.r         = 0.0f;
  camera->color_correction.g         = 0.0f;
  camera->color_correction.b         = 0.0f;
  camera->do_firefly_rejection       = false;
  camera->indirect_only              = false;
  camera->film_grain                 = 0.0f;
  camera->camera_scale               = 1.0f;
  camera->object_distance            = 1.0f;
  camera->use_physical_camera        = false;

  camera->thin_lens.fov           = 1.0f;
  camera->thin_lens.aperture_size = 0.0f;

  camera->physical.allow_reflections      = false;
  camera->physical.use_spectral_rendering = false;

  // Temp - Canon 50mm F1.2 from 1950s
  camera->physical.focal_length          = 50.53f;
  camera->physical.front_focal_point     = -22.69f;
  camera->physical.back_focal_point      = 65.18f;
  camera->physical.front_principal_point = 27.84f;
  camera->physical.back_principal_point  = 14.65f;
  camera->physical.aperture_point        = 28.02f;
  camera->physical.aperture_diameter     = 21.411f;
  camera->physical.exit_pupil_point      = 26.55f;  // These are probably wrong
  camera->physical.exit_pupil_diameter   = 41.67f;  // These are probably wrong
  camera->physical.image_plane_distance  = 65.18f;
  camera->physical.sensor_width          = 35.0f;

  return LUMINARY_SUCCESS;
}

#define __CAMERA_STANDARD_DIRTY(var) \
  {                                  \
    if (input->var != old->var) {    \
      *output_dirty      = true;     \
      *integration_dirty = true;     \
      return LUMINARY_SUCCESS;       \
    }                                \
  }

#define __CAMERA_OUTPUT_DIRTY(var) \
  {                                \
    if (input->var != old->var) {  \
      *output_dirty = true;        \
    }                              \
  }

LuminaryResult camera_check_for_dirty(const Camera* input, const Camera* old, bool* output_dirty, bool* integration_dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(output_dirty);
  __CHECK_NULL_ARGUMENT(integration_dirty);

  *output_dirty      = false;
  *integration_dirty = false;

  __CAMERA_STANDARD_DIRTY(pos.x);
  __CAMERA_STANDARD_DIRTY(pos.y);
  __CAMERA_STANDARD_DIRTY(pos.z);
  __CAMERA_STANDARD_DIRTY(rotation.x);
  __CAMERA_STANDARD_DIRTY(rotation.y);
  __CAMERA_STANDARD_DIRTY(rotation.z);
  __CAMERA_STANDARD_DIRTY(russian_roulette_threshold);
  __CAMERA_STANDARD_DIRTY(camera_scale);
  __CAMERA_STANDARD_DIRTY(object_distance);
  __CAMERA_STANDARD_DIRTY(use_physical_camera);

  __CAMERA_STANDARD_DIRTY(aperture_shape);

  if (input->aperture_shape != LUMINARY_APERTURE_ROUND) {
    __CAMERA_STANDARD_DIRTY(aperture_blade_count);
  }

  __CAMERA_STANDARD_DIRTY(thin_lens.fov);
  __CAMERA_STANDARD_DIRTY(thin_lens.aperture_size);

  __CAMERA_STANDARD_DIRTY(physical.allow_reflections);
  __CAMERA_STANDARD_DIRTY(physical.use_spectral_rendering);
  __CAMERA_STANDARD_DIRTY(physical.focal_length);
  __CAMERA_STANDARD_DIRTY(physical.front_focal_point);
  __CAMERA_STANDARD_DIRTY(physical.back_focal_point);
  __CAMERA_STANDARD_DIRTY(physical.front_principal_point);
  __CAMERA_STANDARD_DIRTY(physical.back_principal_point);
  __CAMERA_STANDARD_DIRTY(physical.aperture_point);
  __CAMERA_STANDARD_DIRTY(physical.aperture_diameter);
  __CAMERA_STANDARD_DIRTY(physical.exit_pupil_point);
  __CAMERA_STANDARD_DIRTY(physical.exit_pupil_diameter);
  __CAMERA_STANDARD_DIRTY(physical.image_plane_distance);
  __CAMERA_STANDARD_DIRTY(physical.sensor_width);

  __CAMERA_OUTPUT_DIRTY(exposure);
  __CAMERA_OUTPUT_DIRTY(bloom_blend);
  __CAMERA_OUTPUT_DIRTY(lens_flare);
  __CAMERA_OUTPUT_DIRTY(lens_flare_threshold);
  __CAMERA_OUTPUT_DIRTY(dithering);
  __CAMERA_OUTPUT_DIRTY(tonemap);
  __CAMERA_OUTPUT_DIRTY(filter);
  __CAMERA_OUTPUT_DIRTY(purkinje);
  __CAMERA_OUTPUT_DIRTY(purkinje_kappa1);
  __CAMERA_OUTPUT_DIRTY(purkinje_kappa2);
  __CAMERA_OUTPUT_DIRTY(use_color_correction);
  __CAMERA_OUTPUT_DIRTY(film_grain);
  __CAMERA_OUTPUT_DIRTY(do_firefly_rejection);
  __CAMERA_OUTPUT_DIRTY(indirect_only);

  if (input->tonemap == LUMINARY_TONEMAP_AGX_CUSTOM) {
    __CAMERA_OUTPUT_DIRTY(agx_custom_slope);
    __CAMERA_OUTPUT_DIRTY(agx_custom_power);
    __CAMERA_OUTPUT_DIRTY(agx_custom_saturation);
  }

  if (input->use_color_correction) {
    __CAMERA_OUTPUT_DIRTY(color_correction.r);
    __CAMERA_OUTPUT_DIRTY(color_correction.g);
    __CAMERA_OUTPUT_DIRTY(color_correction.b);
  }

  return LUMINARY_SUCCESS;
}
