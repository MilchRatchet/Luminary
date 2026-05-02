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
              [0]  = {.radius = -94.29f, .vertex = 0.0f, .cylindrical_radius = 28.0f},
              [1]  = {.radius = 181.58f, .vertex = 7.17f, .cylindrical_radius = 28.0f},
              [2]  = {.radius = -72.86f, .vertex = 9.3f, .cylindrical_radius = 24.0f},
              [3]  = {.radius = 76.74f, .vertex = 21.7f, .cylindrical_radius = 24.0f},
              [4]  = {.radius = -43.02f, .vertex = 23.83f, .cylindrical_radius = 24.0f},
              [5]  = {.radius = 27.44f, .vertex = 45.14f, .cylindrical_radius = 34.0f},
              [6]  = {.radius = -321.70f, .vertex = 49.53f, .cylindrical_radius = 34.0f},
              [7]  = {.radius = 50.96f, .vertex = 70.01f, .cylindrical_radius = 34.0f},
              [8]  = {.radius = 120.34f, .vertex = 70.97f, .cylindrical_radius = 40.0f},
              [9]  = {.radius = 68.99f, .vertex = 78.97f, .cylindrical_radius = 40.0f},
              [10] = {.radius = 251.93f, .vertex = 79.18f, .cylindrical_radius = 46.4f},
              [11] = {.radius = 94.00f, .vertex = 88.18f, .cylindrical_radius = 46.4f},
            },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.6435f, .abbe = 53.5f, .cylindrical_radius = 28.0f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 24.0f},
              [4]  = {.design_ior = 1.5174f, .abbe = 52.5f, .cylindrical_radius = 24.0f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.7174f, .abbe = 29.5f, .cylindrical_radius = 34.0f},
              [7]  = {.design_ior = 1.6385f, .abbe = 55.5f, .cylindrical_radius = 34.0f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.7173f, .abbe = 47.9f, .cylindrical_radius = 40.0f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [11] = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 46.4f},
              [12] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 100.0f,
          .aperture_point      = 16.537354f,
          .last_vertex         = 88.18f,
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
          .design_focal_length = 100.0f,
          .aperture_point      = 15.89f,
          .last_vertex         = 39.14f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_C] =
        {
          .num_interfaces = 5,
          .interfaces =
            {
              [0] = {.radius = -411.4539f, .vertex = 0.0f, .cylindrical_radius = 30.47232f},
              [1] = {.radius = 49.3140f, .vertex = 17.5333f, .cylindrical_radius = 30.47232f},
              [2] = {.radius = 110.1737f, .vertex = 87.4842f, .cylindrical_radius = 30.47232f},
              [3] = {.radius = 101.6586f, .vertex = 338.1497f, .cylindrical_radius = 91.41696f},
              [4] = {.radius = FLT_MAX, .vertex = 353.1782f, .cylindrical_radius = 157.4403f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.67025f, .abbe = 57.53f, .cylindrical_radius = 30.47232f},
              [2] = {.design_ior = 1.80518f, .abbe = 25.35f, .cylindrical_radius = 30.47232f},
              [3] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [4] = {.design_ior = 1.62041f, .abbe = 60.14f, .cylindrical_radius = 157.4403f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 100.0f,
          .aperture_point      = 353.1782f - 353.18f,
          .last_vertex         = 353.1782f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_D] =
        {
          .num_interfaces = 5,
          .interfaces =
            {
              [0] = {.radius = -14.0181f, .vertex = 0.0f, .cylindrical_radius = 9.16666667f},
              [1] = {.radius = -10.8276f, .vertex = 5.6666667f, .cylindrical_radius = 8.875f},
              [2] = {.radius = 129.1860f, .vertex = 44.83333334f, .cylindrical_radius = 12.08333334f},
              [3] = {.radius = -78.9097f, .vertex = 46.0f, .cylindrical_radius = 12.08333334f},
              [4] = {.radius = 26.9147f, .vertex = 51.166666667f, .cylindrical_radius = 12.08333334f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.58913f, .abbe = 61.09f, .cylindrical_radius = 9.16666667f},
              [2] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.80384f, .abbe = 33.89f, .cylindrical_radius = 12.08333334f},
              [4] = {.design_ior = 1.58913f, .abbe = 61.09f, .cylindrical_radius = 12.08333334f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 100.0f,
          .aperture_point      = 51.166666667f - 23.0f,
          .last_vertex         = 51.166666667f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_E] =
        {
          .num_interfaces = 9,
          {
            [0] = {.radius = -190.62f, .vertex = 0.000f, .cylindrical_radius = 30.0f},
            [1] = {.radius = 59.39f, .vertex = 13.72f, .cylindrical_radius = 30.0f},
            [2] = {.radius = 29.03f, .vertex = 36.66f, .cylindrical_radius = 42.0f},
            [3] = {.radius = -105.16f, .vertex = 40.05f, .cylindrical_radius = 42.0f},
            [4] = {.radius = 50.42f, .vertex = 73.75f, .cylindrical_radius = 42.0f},
            [5] = {.radius = 118.55f, .vertex = 74.25f, .cylindrical_radius = 49.0f},
            [6] = {.radius = 073.27f, .vertex = 82.63f, .cylindrical_radius = 49.0f},
            [7] = {.radius = 310.7f, .vertex = 83.13f, .cylindrical_radius = 52.0f},
            [8] = {.radius = 121.47f, .vertex = 91.94f, .cylindrical_radius = 52.0f},
          },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 30.0f},
              [2] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.6945f, .abbe = 30.7f, .cylindrical_radius = 42.0f},
              [4] = {.design_ior = 1.6062f, .abbe = 59.8f, .cylindrical_radius = 42.0f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 49.0f},
              [7] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [8] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 52.0f},
              [9] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 100.0f,
          .aperture_point      = 91.94f - 71.56f,
          .last_vertex         = 91.94f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_F] =
        {
          .num_interfaces = 4,
          {
            [0] = {.radius = -337.378f, .vertex = 0.000f, .cylindrical_radius = 8.6875f},
            [1] = {.radius = -32.0041f, .vertex = 1.2f, .cylindrical_radius = 8.6875f},
            [2] = {.radius = 32.0041f, .vertex = 3.7f, .cylindrical_radius = 8.6875f},
            [3] = {.radius = 51.416f, .vertex = 4.9f, .cylindrical_radius = 8.6875f},
          },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.57125f, .abbe = 55.8f, .cylindrical_radius = 8.6875f},
              [2] = {.design_ior = 1.54408, .abbe = 73.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.67245f, .abbe = 45.8f, .cylindrical_radius = 8.6875f},
              [4] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 100.0f,
          .aperture_point      = 4.9f - 69.5f,
          .last_vertex         = 4.9f,
        },
};

LuminaryResult lens_library_get_template_data(LuminaryLensTemplate template, LensTemplateData* data) {
  __CHECK_NULL_ARGUMENT(data);

  if (template >= LUMINARY_LENS_TEMPLATE_COUNT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Specified invalid lens template.");

  memcpy(data, _template_data + template, sizeof(LensTemplateData));

  return LUMINARY_SUCCESS;
}
