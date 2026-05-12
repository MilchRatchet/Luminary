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
              [0]  = {.radius = -47.145f, .vertex = 0.0f, .cylindrical_radius = 13.75f},
              [1]  = {.radius = 90.79f, .vertex = 3.585f, .cylindrical_radius = 13.75f},
              [2]  = {.radius = -36.43f, .vertex = 4.65f, .cylindrical_radius = 11.96f},
              [3]  = {.radius = 38.37f, .vertex = 10.85f, .cylindrical_radius = 11.96f},
              [4]  = {.radius = -21.51f, .vertex = 11.915f, .cylindrical_radius = 10.835f},
              [5]  = {.radius = 13.72f, .vertex = 22.57f, .cylindrical_radius = 11.275f},
              [6]  = {.radius = -160.85f, .vertex = 24.765f, .cylindrical_radius = 17.49f},
              [7]  = {.radius = 25.48f, .vertex = 35.005f, .cylindrical_radius = 17.49f},
              [8]  = {.radius = 60.17f, .vertex = 35.485f, .cylindrical_radius = 20.31f},
              [9]  = {.radius = 34.495f, .vertex = 39.485f, .cylindrical_radius = 20.31f},
              [10] = {.radius = 125.965f, .vertex = 39.59f, .cylindrical_radius = 23.33f},
              [11] = {.radius = 47.00f, .vertex = 44.09f, .cylindrical_radius = 23.33f},
            },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.6435f, .abbe = 53.5f, .cylindrical_radius = 13.75f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 11.96f},
              [4]  = {.design_ior = 1.5174f, .abbe = 52.5f, .cylindrical_radius = 11.96f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.7174f, .abbe = 29.5f, .cylindrical_radius = 17.49f},
              [7]  = {.design_ior = 1.6385f, .abbe = 55.5f, .cylindrical_radius = 17.49f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.7173f, .abbe = 47.9f, .cylindrical_radius = 20.31f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [11] = {.design_ior = 1.6935f, .abbe = 53.5f, .cylindrical_radius = 23.33f},
              [12] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 8.268677f,
          .last_vertex         = 44.09f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_B] =
        {
          .num_interfaces = 7,
          .interfaces =
            {
              [0] = {.radius = -24.523f, .vertex = 0.0f, .cylindrical_radius = 8.66f},
              [1] = {.radius = 16.4835f, .vertex = 5.0f, .cylindrical_radius = 8.66f},
              [2] = {.radius = 381.775f, .vertex = 6.195f, .cylindrical_radius = 8.29f},
              [3] = {.radius = 16.8175f, .vertex = 9.92f, .cylindrical_radius = 7.225f},
              [4] = {.radius = -35.52f, .vertex = 11.37f, .cylindrical_radius = 8.69f},
              [5] = {.radius = FLT_MAX, .vertex = 14.42f, .cylindrical_radius = 9.885f},
              [6] = {.radius = 19.655f, .vertex = 19.57f, .cylindrical_radius = 9.885f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.66080f, .abbe = 50.8f, .cylindrical_radius = 8.66f},
              [2] = {.design_ior = 1.57380f, .abbe = 42.5f, .cylindrical_radius = 8.66f},
              [3] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [4] = {.design_ior = 1.61200f, .abbe = 37.2f, .cylindrical_radius = 8.69f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6] = {.design_ior = 1.67786f, .abbe = 55.5f, .cylindrical_radius = 9.885f},
              [7] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 7.945f,
          .last_vertex         = 19.57f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_C] =
        {
          .num_interfaces = 5,
          .interfaces =
            {
              [0] = {.radius = -82.29078f, .vertex = 0.0f, .cylindrical_radius = 6.312f},
              [1] = {.radius = 9.8628f, .vertex = 3.50666f, .cylindrical_radius = 6.312f},
              [2] = {.radius = 22.03474f, .vertex = 17.49684f, .cylindrical_radius = 6.312f},
              [3] = {.radius = 20.33172f, .vertex = 67.62994f, .cylindrical_radius = 18.72f},
              [4] = {.radius = FLT_MAX, .vertex = 70.63564f, .cylindrical_radius = 31.23f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.67025f, .abbe = 57.53f, .cylindrical_radius = 6.312f},
              [2] = {.design_ior = 1.80518f, .abbe = 25.35f, .cylindrical_radius = 6.312f},
              [3] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [4] = {.design_ior = 1.62041f, .abbe = 60.14f, .cylindrical_radius = 31.23f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 20.0f,
          .aperture_point      = 70.63564f - 70.636f,
          .last_vertex         = 70.63564f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_D] =
        {
          .num_interfaces = 5,
          .interfaces =
            {
              [0] = {.radius = -16.82172f, .vertex = 0.0f, .cylindrical_radius = 12.205f},
              [1] = {.radius = -12.99312f, .vertex = 6.8f, .cylindrical_radius = 11.065f},
              [2] = {.radius = 155.0232f, .vertex = 53.8f, .cylindrical_radius = 12.46f},
              [3] = {.radius = -94.69164f, .vertex = 55.2f, .cylindrical_radius = 14.66f},
              [4] = {.radius = 32.29764f, .vertex = 61.4f, .cylindrical_radius = 14.66f},
            },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.58913f, .abbe = 61.09f, .cylindrical_radius = 12.205f},
              [2] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.80384f, .abbe = 33.89f, .cylindrical_radius = 14.66f},
              [4] = {.design_ior = 1.58913f, .abbe = 61.09f, .cylindrical_radius = 14.66f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 120.0f,
          .aperture_point      = 33.8f,
          .last_vertex         = 61.4f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_E] =
        {
          .num_interfaces = 9,
          {
            [0] = {.radius = -190.62f, .vertex = 0.000f, .cylindrical_radius = 30.18f},
            [1] = {.radius = 59.39f, .vertex = 13.72f, .cylindrical_radius = 30.18f},
            [2] = {.radius = 29.03f, .vertex = 36.66f, .cylindrical_radius = 24.64f},
            [3] = {.radius = -105.16f, .vertex = 40.05f, .cylindrical_radius = 42.83f},
            [4] = {.radius = 50.42f, .vertex = 73.75f, .cylindrical_radius = 42.83f},
            [5] = {.radius = 118.55f, .vertex = 74.25f, .cylindrical_radius = 49.275f},
            [6] = {.radius = 073.27f, .vertex = 82.63f, .cylindrical_radius = 49.275f},
            [7] = {.radius = 310.7f, .vertex = 83.13f, .cylindrical_radius = 52.24f},
            [8] = {.radius = 121.47f, .vertex = 91.94f, .cylindrical_radius = 52.24f},
          },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 30.18f},
              [2] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.6945f, .abbe = 30.7f, .cylindrical_radius = 42.83f},
              [4] = {.design_ior = 1.6062f, .abbe = 59.8f, .cylindrical_radius = 42.83f},
              [5] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 49.275f},
              [7] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [8] = {.design_ior = 1.6135f, .abbe = 59.4f, .cylindrical_radius = 52.24f},
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
            [0] = {.radius = -2699.024f, .vertex = 0.000f, .cylindrical_radius = 69.815f},
            [1] = {.radius = -256.0328f, .vertex = 9.6f, .cylindrical_radius = 69.815f},
            [2] = {.radius = 256.0328f, .vertex = 29.6f, .cylindrical_radius = 69.815f},
            [3] = {.radius = 411.328f, .vertex = 39.2f, .cylindrical_radius = 69.815f},
          },
          .media =
            {
              [0] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1] = {.design_ior = 1.57125f, .abbe = 55.8f, .cylindrical_radius = 69.815f},
              [2] = {.design_ior = 1.54408f, .abbe = 73.0f, .cylindrical_radius = FLT_MAX},
              [3] = {.design_ior = 1.67245f, .abbe = 45.8f, .cylindrical_radius = 69.815f},
              [4] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 800.0f,
          .aperture_point      = 39.2f - 556.0f,
          .last_vertex         = 39.2f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_G] =
        {
          .num_interfaces = 10,
          {
            [0] = {.radius = -60.00f, .vertex = 0.000f, .cylindrical_radius = 14.35f},
            [1] = {.radius = 53.00f, .vertex = 6.95f, .cylindrical_radius = 14.35f},
            [2] = {.radius = -26.60f, .vertex = 7.435f, .cylindrical_radius = 14.24f},
            [3] = {.radius = 25.25f, .vertex = 18.045f, .cylindrical_radius = 14.24f},
            [4] = {.radius = -19.25f, .vertex = 20.57f, .cylindrical_radius = 11.255f},
            [5] = {.radius = 14.15f, .vertex = 30.02f, .cylindrical_radius = 11.285f},
            [6] = {.radius = -575.00f, .vertex = 32.545f, .cylindrical_radius = 16.09f},
            [7] = {.radius = 22.40f, .vertex = 40.32f, .cylindrical_radius = 16.09f},
            [8] = {.radius = 160.50f, .vertex = 41.145f, .cylindrical_radius = 18.04f},
            [9] = {.radius = 41.80f, .vertex = 46.52f, .cylindrical_radius = 18.04f},
          },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.64238f, .abbe = 48.0f, .cylindrical_radius = 14.35f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.64238f, .abbe = 48.0f, .cylindrical_radius = 14.24f},
              [4]  = {.design_ior = 1.67270f, .abbe = 32.2f, .cylindrical_radius = 14.24f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.57566f, .abbe = 41.2f, .cylindrical_radius = 16.09f},
              [7]  = {.design_ior = 1.62306f, .abbe = 56.9f, .cylindrical_radius = 16.09f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.64238f, .abbe = 48.0f, .cylindrical_radius = 18.04f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 46.52f - 22.22f,
          .last_vertex         = 46.52f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_H] =
        {
          .num_interfaces = 10,
          {
            [0] = {.radius = -51.565f, .vertex = 0.000f, .cylindrical_radius = 11.83f},
            [1] = {.radius = -11.060f, .vertex = 2.285f, .cylindrical_radius = 10.04f},
            [2] = {.radius = 25.545f, .vertex = 12.190f, .cylindrical_radius = 10.04f},
            [3] = {.radius = FLT_MAX, .vertex = 13.430f, .cylindrical_radius = 10.04f},
            [4] = {.radius = 11.755f, .vertex = 21.050f, .cylindrical_radius = 9.78f},
            [5] = {.radius = -323.155f, .vertex = 22.000f, .cylindrical_radius = 14.50f},
            [6] = {.radius = 42.890f, .vertex = 25.525f, .cylindrical_radius = 14.50f},
            [7] = {.radius = 17.930f, .vertex = 31.430f, .cylindrical_radius = 14.50f},
            [8] = {.radius = 216.920f, .vertex = 31.620f, .cylindrical_radius = 16.675f},
            [9] = {.radius = 34.605f, .vertex = 36.285f, .cylindrical_radius = 16.675f},
          },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.5894f, .abbe = 51.2f, .cylindrical_radius = 11.83f},
              [2]  = {.design_ior = 1.6578f, .abbe = 51.2f, .cylindrical_radius = 10.04f},
              [3]  = {.design_ior = 1.5232f, .abbe = 50.9f, .cylindrical_radius = 10.04f},
              [4]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [5]  = {.design_ior = 1.7394f, .abbe = 28.2f, .cylindrical_radius = 9.78f},
              [6]  = {.design_ior = 1.4962f, .abbe = 70.1f, .cylindrical_radius = 14.50f},
              [7]  = {.design_ior = 1.6710f, .abbe = 47.2f, .cylindrical_radius = 14.50f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.6710f, .abbe = 47.2f, .cylindrical_radius = 16.675f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 14.56f,
          .last_vertex         = 36.285f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_I] =
        {
          .num_interfaces = 13,
          {
            [0]  = {.radius = -137.0000f, .vertex = 0.000f, .cylindrical_radius = 16.06f},
            [1]  = {.radius = 118.3850f, .vertex = 3.1750f, .cylindrical_radius = 16.06f},
            [2]  = {.radius = -26.9750f, .vertex = 3.3200f, .cylindrical_radius = 16.20f},
            [3]  = {.radius = -63.9250f, .vertex = 7.6500f, .cylindrical_radius = 16.20f},
            [4]  = {.radius = -25.7850f, .vertex = 7.7950f, .cylindrical_radius = 16.20f},
            [5]  = {.radius = -222.2500f, .vertex = 14.7250f, .cylindrical_radius = 16.20f},
            [6]  = {.radius = -15.4300f, .vertex = 15.8800f, .cylindrical_radius = 11.56f},
            [7]  = {.radius = 14.6050f, .vertex = 31.8550f, .cylindrical_radius = 11.845f},
            [8]  = {.radius = 22.8750f, .vertex = 34.3550f, .cylindrical_radius = 14.50f},
            [9]  = {.radius = 25.1500f, .vertex = 35.3650f, .cylindrical_radius = 15.32f},
            [10] = {.radius = 24.0250f, .vertex = 40.1800f, .cylindrical_radius = 17.45f},
            [11] = {.radius = 115.6650f, .vertex = 40.3250f, .cylindrical_radius = 22.50f},
            [12] = {.radius = 35.6900f, .vertex = 46.6750f, .cylindrical_radius = 22.50f},
          },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.7762f, .abbe = 49.4f, .cylindrical_radius = 16.06f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.7762f, .abbe = 49.4f, .cylindrical_radius = 16.20f},
              [4]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [5]  = {.design_ior = 1.7762f, .abbe = 49.4f, .cylindrical_radius = 16.20f},
              [6]  = {.design_ior = 1.7617f, .abbe = 27.3f, .cylindrical_radius = 11.56f},
              [7]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [8]  = {.design_ior = 1.7462f, .abbe = 27.9f, .cylindrical_radius = 14.50f},
              [9]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [10] = {.design_ior = 1.7007f, .abbe = 46.7f, .cylindrical_radius = 17.45f},
              [11] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [12] = {.design_ior = 1.6810f, .abbe = 54.7f, .cylindrical_radius = 22.50f},
              [13] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 31.8550f - 8.8750f,
          .last_vertex         = 46.6750f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_J] =
        {
          .num_interfaces = 10,
          {
            [0] = {.radius = -48.00f, .vertex = 0.00f, .cylindrical_radius = 12.105f},
            [1] = {.radius = 113.65f, .vertex = 4.75f, .cylindrical_radius = 12.105f},
            [2] = {.radius = -21.40f, .vertex = 5.05f, .cylindrical_radius = 10.67f},
            [3] = {.radius = 68.50f, .vertex = 12.10f, .cylindrical_radius = 10.67f},
            [4] = {.radius = -15.05f, .vertex = 14.00f, .cylindrical_radius = 8.76f},
            [5] = {.radius = 13.35f, .vertex = 23.95f, .cylindrical_radius = 9.59f},
            [6] = {.radius = 26.75f, .vertex = 28.15f, .cylindrical_radius = 12.305f},
            [7] = {.radius = 21.75f, .vertex = 31.50f, .cylindrical_radius = 12.305f},
            [8] = {.radius = 87.45f, .vertex = 31.60f, .cylindrical_radius = 14.405f},
            [9] = {.radius = 28.95f, .vertex = 36.20f, .cylindrical_radius = 14.405f},
          },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.66200f, .abbe = 56.1f, .cylindrical_radius = 12.105f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.66200f, .abbe = 56.1f, .cylindrical_radius = 10.67f},
              [4]  = {.design_ior = 1.60156f, .abbe = 35.2f, .cylindrical_radius = 8.76f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.67246f, .abbe = 32.3f, .cylindrical_radius = 9.59f},
              [7]  = {.design_ior = 1.66200f, .abbe = 56.1f, .cylindrical_radius = 12.305f},
              [8]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [9]  = {.design_ior = 1.66200f, .abbe = 56.1f, .cylindrical_radius = 14.405f},
              [10] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 50.0f,
          .aperture_point      = 18.65f,
          .last_vertex         = 36.20f,
        },
      [LUMINARY_LENS_TEMPLATE_PHYSICAL_K] =
        {
          .num_interfaces = 11,
          {
            [0]  = {.radius = -58.807f, .vertex = 0.000f, .cylindrical_radius = 14.84f},
            [1]  = {.radius = -16.510f, .vertex = 8.806f, .cylindrical_radius = 9.24f},
            [2]  = {.radius = -27.517f, .vertex = 11.007f, .cylindrical_radius = 9.25f},
            [3]  = {.radius = 14.308f, .vertex = 24.216f, .cylindrical_radius = 5.24f},
            [4]  = {.radius = 146.160f, .vertex = 24.877f, .cylindrical_radius = 5.24f},
            [5]  = {.radius = 9.079f, .vertex = 27.519f, .cylindrical_radius = 4.875f},
            [6]  = {.radius = FLT_MAX, .vertex = 28.180f, .cylindrical_radius = 7.565f},
            [7]  = {.radius = 22.011f, .vertex = 30.007f, .cylindrical_radius = 7.565f},
            [8]  = {.radius = 11.665f, .vertex = 31.922f, .cylindrical_radius = 7.565f},
            [9]  = {.radius = 55.030f, .vertex = 32.472f, .cylindrical_radius = 10.975f},
            [10] = {.radius = 18.711f, .vertex = 36.983f, .cylindrical_radius = 10.975f},
          },
          .media =
            {
              [0]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [1]  = {.design_ior = 1.5333f, .abbe = 48.9f, .cylindrical_radius = 14.84f},
              [2]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [3]  = {.design_ior = 1.6716f, .abbe = 47.2f, .cylindrical_radius = 9.25f},
              [4]  = {.design_ior = 1.4645f, .abbe = 65.7f, .cylindrical_radius = 5.24f},
              [5]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [6]  = {.design_ior = 1.6890f, .abbe = 31.0f, .cylindrical_radius = 4.875f},
              [7]  = {.design_ior = 1.4645f, .abbe = 65.7f, .cylindrical_radius = 7.565f},
              [8]  = {.design_ior = 1.6716f, .abbe = 47.2f, .cylindrical_radius = 7.565f},
              [9]  = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
              [10] = {.design_ior = 1.6716f, .abbe = 47.2f, .cylindrical_radius = 10.975f},
              [11] = {.design_ior = IOR_AIR, .abbe = 0.0f, .cylindrical_radius = FLT_MAX},
            },
          .design_focal_length = 35.0f,
          .aperture_point      = 25.677f,
          .last_vertex         = 36.983f,
        },
};

LuminaryResult lens_library_get_template_data(LuminaryLensTemplate template, LensTemplateData* data) {
  __CHECK_NULL_ARGUMENT(data);

  if (template >= LUMINARY_LENS_TEMPLATE_COUNT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Specified invalid lens template.");

  memcpy(data, _template_data + template, sizeof(LensTemplateData));

  return LUMINARY_SUCCESS;
}
