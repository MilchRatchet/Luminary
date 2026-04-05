#include "lens_library.h"

#include <float.h>
#include <string.h>

#include "internal_error.h"

static const LensTemplateData
  _template_data[LUMINARY_LENS_TEMPLATE_COUNT] =
    {
      [LUMINARY_LENS_TEMPLATE_THIN_LENS] = {.num_interfaces = 0},
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
          .design_focal_length = 100.0f,
          .aperture_point      = 16.537354f,
          .exit_pupil_point    = 0.0f,
          .exit_pupil_diameter = 28.0f,
        },
};

struct LensTemplateInternalData {
  float design_focal_length;
} typedef LensTemplateInternalData;

static const LensTemplateDefaults _template_defaults[LUMINARY_LENS_TEMPLATE_COUNT] = {
  [LUMINARY_LENS_TEMPLATE_THIN_LENS] =
    {
      .aperture_diameter = 0.0f,
      .focal_length      = 10.0f,
      .sensor_distance   = 10.0f,
    },
  [LUMINARY_LENS_TEMPLATE_PHYSICAL_A] =
    {
      .focal_length      = 50.53f,
      .aperture_diameter = 42.822f,
      .sensor_distance   = 20.622646f,
    },
};

LuminaryResult lens_library_get_template_data(LuminaryLensTemplate template, float focal_length, LensTemplateData* data) {
  __CHECK_NULL_ARGUMENT(data);

  if (template >= LUMINARY_LENS_TEMPLATE_COUNT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Specified invalid lens template.");

  memcpy(data, _template_data + template, sizeof(LensTemplateData));

  if (template == LUMINARY_LENS_TEMPLATE_THIN_LENS)
    return LUMINARY_SUCCESS;

  const float scale = focal_length / data->design_focal_length;
  for (uint32_t interface_id = 0; interface_id < data->num_interfaces; interface_id++) {
    data->interfaces[interface_id].radius *= scale;
    data->interfaces[interface_id].vertex *= scale;
  }

  data->aperture_point *= scale;
  data->exit_pupil_point *= scale;

  return LUMINARY_SUCCESS;
}

LuminaryResult lens_library_get_template_defaults(LuminaryLensTemplate template, LensTemplateDefaults* defaults) {
  __CHECK_NULL_ARGUMENT(defaults);

  if (template >= LUMINARY_LENS_TEMPLATE_COUNT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Specified invalid lens template.");

  memcpy(defaults, _template_defaults + template, sizeof(LensTemplateDefaults));

  return LUMINARY_SUCCESS;
}
