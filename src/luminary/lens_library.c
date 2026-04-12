#include "lens_library.h"

#include <float.h>
#include <string.h>

#include "internal_error.h"

static const LensTemplateData
  _template_data[LUMINARY_LENS_TEMPLATE_COUNT] =
    {
      [LUMINARY_LENS_TEMPLATE_THIN_LENS] =
        {
          .num_interfaces      = 0,
          .design_focal_length = 100.0f,
          .aperture_point      = 0.0f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_A] =
        {
          .num_interfaces = 12,
          .interfaces =
            {
              [0]  = {.radius = -94.29f, .vertex = 0.0f, .cylindrical_radius = 14.0f},
              [1]  = {.radius = 181.58f, .vertex = 7.17f, .cylindrical_radius = 14.0f},
              [2]  = {.radius = -72.86f, .vertex = 9.3f, .cylindrical_radius = 12.0f},
              [3]  = {.radius = 76.74f, .vertex = 21.7f, .cylindrical_radius = 12.0f},
              [4]  = {.radius = -43.02f, .vertex = 23.83f, .cylindrical_radius = 12.0f},
              [5]  = {.radius = 27.44f, .vertex = 45.14f, .cylindrical_radius = 17.0f},
              [6]  = {.radius = -321.70f, .vertex = 49.53f, .cylindrical_radius = 17.0f},
              [7]  = {.radius = 50.96f, .vertex = 70.01f, .cylindrical_radius = 17.0f},
              [8]  = {.radius = 120.34f, .vertex = 70.97f, .cylindrical_radius = 20.0f},
              [9]  = {.radius = 68.99f, .vertex = 78.97f, .cylindrical_radius = 20.0f},
              [10] = {.radius = 251.93f, .vertex = 79.18f, .cylindrical_radius = 23.2f},
              [11] = {.radius = 94.00f, .vertex = 88.18f, .cylindrical_radius = 23.2f},
            },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.6435f, .abbe = 53.5f, .cylindrical_radius = 14.0f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 12.0f},
              [4]  = {.design_ior = 1.5174f, .abbe = 52.5f, .cylindrical_radius = 12.0f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.7174f, .abbe = 29.5f, .cylindrical_radius = 17.0f},
              [7]  = {.design_ior = 1.6385f, .abbe = 55.5f, .cylindrical_radius = 17.0f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.7173f, .abbe = 47.9f, .cylindrical_radius = 20.0f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [11] = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 23.2f},
              [12] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length   = 100.0f,
          .aperture_point        = 16.537354f,
          .exit_pupil_point      = 0.0f,
          .exit_pupil_diameter   = 28.0f,
          .last_vertex           = 88.18f,
          .front_principal_plane = 88.18f - 55.68f,
          .back_principal_plane  = 88.18f - 29.30f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_B] =
        {
          .num_interfaces = 7,
          .interfaces =
            {
              [0] = {.radius = -49.046f, .vertex = 0.0f, .cylindrical_radius = 17.2f},
              [1] = {.radius = 32.967f, .vertex = 10.0f, .cylindrical_radius = 17.2f},
              [2] = {.radius = 763.55f, .vertex = 12.39f, .cylindrical_radius = 17.2f},
              [3] = {.radius = 33.635f, .vertex = 19.84f, .cylindrical_radius = 17.2f},
              [4] = {.radius = -71.04f, .vertex = 22.74f, .cylindrical_radius = 17.2f},
              [5] = {.radius = FLT_MAX, .vertex = 28.84f, .cylindrical_radius = 19.6f},
              [6] = {.radius = 39.31f, .vertex = 39.14f, .cylindrical_radius = 19.6f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.66080f, .abbe = 50.8f, .cylindrical_radius = 17.2f},
              [2] = {.design_ior = 1.57380f, .abbe = 42.5f, .cylindrical_radius = 17.2f},
              [3] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [4] = {.design_ior = 1.61200f, .abbe = 37.2f, .cylindrical_radius = 17.2f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6] = {.design_ior = 1.67786f, .abbe = 55.5f, .cylindrical_radius = 19.6f},
              [7] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length   = 100.0f,
          .aperture_point        = 15.89f,
          .exit_pupil_point      = 0,
          .exit_pupil_diameter   = 34.4f,
          .last_vertex           = 39.14f,
          .front_principal_plane = 39.14f - 18.32f,
          .back_principal_plane  = 39.14f - 19.80f,
        },
};

LuminaryResult lens_library_get_template_data(LuminaryLensTemplate template, LensTemplateData* data) {
  __CHECK_NULL_ARGUMENT(data);

  if (template >= LUMINARY_LENS_TEMPLATE_COUNT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Specified invalid lens template.");

  memcpy(data, _template_data + template, sizeof(LensTemplateData));

  return LUMINARY_SUCCESS;
}
