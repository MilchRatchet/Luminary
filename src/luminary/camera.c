#include "camera.h"

#include "internal_error.h"
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
  camera->filter                       = LUMINARY_FILTER_NONE;
  camera->wasd_speed                   = 1.0f;
  camera->mouse_speed                  = 1.0f;
  camera->smooth_movement              = 0;
  camera->smoothing_factor             = 0.1f;
  camera->purkinje                     = 1;
  camera->purkinje_kappa1              = 0.2f;
  camera->purkinje_kappa2              = 0.29f;
  camera->russian_roulette_threshold   = 0.1f;
  camera->use_color_correction         = 0;
  camera->color_correction.r           = 0.0f;
  camera->color_correction.g           = 0.0f;
  camera->color_correction.b           = 0.0f;
  camera->film_grain                   = 0.0f;
  camera->camera_scale                 = 1.0f;
  camera->object_distance              = 1.0f;
  camera->use_physical_camera          = false;

  camera->thin_lens.fov           = 1.0f;
  camera->thin_lens.aperture_size = 0.0f;

  camera->physical.allow_reflections      = false;
  camera->physical.use_spectral_rendering = false;

  // Temp - Canon 50mm F1.2 from 1950s
  const float scale             = 50.53f / 100.0f;
  const float last_vertex_point = 88.18f * scale;

  camera->physical.focal_length          = 50.53f;
  camera->physical.front_focal_point     = last_vertex_point - (-22.69f);
  camera->physical.back_focal_point      = last_vertex_point - 65.18f;
  camera->physical.front_principal_point = last_vertex_point - 27.84f;
  camera->physical.back_principal_point  = last_vertex_point - 14.65f;
  camera->physical.aperture_point        = last_vertex_point - 28.02f;
  camera->physical.aperture_diameter     = 21.411f;
  camera->physical.exit_pupil_point      = 0.0f;   // last_vertex_point - 26.55f;
  camera->physical.exit_pupil_diameter   = 28.0f;  // 34.64f;
  camera->physical.image_plane_distance  = 65.18f - last_vertex_point;
  camera->physical.sensor_width          = 20.0f;

  return LUMINARY_SUCCESS;
}

#define CAMERA_ALL_DIRTY_FLAGS ((uint32_t) (SCENE_DIRTY_FLAG_CAMERA | SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT))

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
  __CAMERA_CHECK_DIRTY(russian_roulette_threshold, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(camera_scale, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(object_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(use_physical_camera, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  __CAMERA_CHECK_DIRTY(aperture_shape, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);

  if (input->aperture_shape != LUMINARY_APERTURE_ROUND) {
    __CAMERA_CHECK_DIRTY(aperture_blade_count, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  if (input->use_physical_camera) {
    __CAMERA_CHECK_DIRTY(physical.allow_reflections, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.use_spectral_rendering, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.focal_length, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.front_focal_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.back_focal_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.front_principal_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.back_principal_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.aperture_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.aperture_diameter, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.exit_pupil_point, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.exit_pupil_diameter, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.image_plane_distance, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(physical.sensor_width, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }
  else {
    __CAMERA_CHECK_DIRTY(thin_lens.fov, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
    __CAMERA_CHECK_DIRTY(thin_lens.aperture_size, SCENE_DIRTY_FLAG_INTEGRATION | SCENE_DIRTY_FLAG_OUTPUT);
  }

  __CAMERA_CHECK_DIRTY(use_local_error_minimization, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(exposure, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(bloom_blend, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(dithering, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(tonemap, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(filter, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa1, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(purkinje_kappa2, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(use_color_correction, SCENE_DIRTY_FLAG_OUTPUT);
  __CAMERA_CHECK_DIRTY(film_grain, SCENE_DIRTY_FLAG_OUTPUT);

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
