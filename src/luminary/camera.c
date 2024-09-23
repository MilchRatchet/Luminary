#include "camera.h"

#include "internal_error.h"
#include "utils.h"

LuminaryResult camera_get_default(Camera* camera) {
  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera was NULL.");
  }

  camera->pos.x                      = 0.0f;
  camera->pos.y                      = 0.0f;
  camera->pos.z                      = 0.0f;
  camera->rotation.x                 = 0.0f;
  camera->rotation.y                 = 0.0f;
  camera->rotation.z                 = 0.0f;
  camera->fov                        = 1.0f;
  camera->focal_length               = 1.0f;
  camera->aperture_size              = 0.0f;
  camera->aperture_shape             = LUMINARY_APERTURE_ROUND;
  camera->aperture_blade_count       = 7;
  camera->exposure                   = 1.0f;
  camera->min_exposure               = 10.0f;
  camera->max_exposure               = 400.0f;
  camera->auto_exposure              = 0;
  camera->bloom                      = 1;
  camera->bloom_blend                = 0.01f;
  camera->lens_flare                 = 0;
  camera->lens_flare_threshold       = 1.0f;
  camera->dithering                  = 1;
  camera->far_clip_distance          = 50000.0f;
  camera->tonemap                    = LUMINARY_TONEMAP_AGX;
  camera->agx_custom_slope           = 1.0f;
  camera->agx_custom_power           = 1.0f;
  camera->agx_custom_saturation      = 1.0f;
  camera->filter                     = LUMINARY_FILTER_NONE;
  camera->wasd_speed                 = 1.0f;
  camera->mouse_speed                = 1.0f;
  camera->smooth_movement            = 0;
  camera->smoothing_factor           = 0.1f;
  camera->temporal_blend_factor      = 0.15f;
  camera->purkinje                   = 1;
  camera->purkinje_kappa1            = 0.2f;
  camera->purkinje_kappa2            = 0.29f;
  camera->russian_roulette_threshold = 0.1f;
  camera->use_color_correction       = 0;
  camera->color_correction.r         = 0.0f;
  camera->color_correction.g         = 0.0f;
  camera->color_correction.b         = 0.0f;
  camera->do_firefly_clamping        = 1;
  camera->film_grain                 = 0.0f;

  return LUMINARY_SUCCESS;
}

LuminaryResult camera_check_for_dirty(const Camera* a, const Camera* b, bool* is_dirty) {
  LUM_UNUSED(a);
  LUM_UNUSED(b);
  LUM_UNUSED(is_dirty);
  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}
