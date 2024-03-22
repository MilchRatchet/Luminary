#ifndef CU_BRDF_UNITTEST_H
#define CU_BRDF_UNITTEST_H

#include "bsdf.cuh"
#include "buffer.h"
#include "utils.cuh"
#include "utils.h"

#define BRDF_UNITTEST_STEPS_SMOOTHNESS 20
#define BRDF_UNITTEST_STEPS_METALLIC 2
#define BRDF_UNITTEST_TOTAL (BRDF_UNITTEST_STEPS_SMOOTHNESS * BRDF_UNITTEST_STEPS_METALLIC)
#define BRDF_UNITTEST_ITERATIONS (8192 * 32)

// TODO:
// This is broken because the random number generator isn't used correctly in this context.
// The solution would be a macro based system that is setup before any of the includes.
// However, for that the unittests would need to be in a separate translation unit.

LUMINARY_KERNEL void brdf_unittest_kernel(float* bounce, float* light) {
  unsigned int id = THREAD_ID;

  const unsigned int total = BRDF_UNITTEST_TOTAL;

  while (id < total) {
    const unsigned int x = id % BRDF_UNITTEST_STEPS_SMOOTHNESS;
    const unsigned int y = id / BRDF_UNITTEST_STEPS_SMOOTHNESS;

    const float smoothness = (1.0f / (BRDF_UNITTEST_STEPS_SMOOTHNESS - 1)) * x;
    const float metallic   = (1.0f / (BRDF_UNITTEST_STEPS_METALLIC - 1)) * y;

    float sum_bounce = 0.0f;
    float sum_light  = 0.0f;

    GBufferData data;
    data.albedo    = get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f);
    data.position  = get_vector(FLT_MAX, 1000000.0f, 0.0f);
    data.ior_in    = 1.0f;
    data.ior_out   = 1.0f;
    data.roughness = 1.0f - smoothness;
    data.metallic  = metallic;
    data.normal    = get_vector(0.0f, 1.0f, 0.0f);

    for (int i = 0; i < BRDF_UNITTEST_ITERATIONS; i++) {
      float2 ran0 = quasirandom_sequence_2D_base(0, make_ushort2(0, 0), i, 0);
      float2 ran1 = quasirandom_sequence_2D_base(1, make_ushort2(0, 0), i, 0);

      ran0.x = 0.5f * PI * (1.0f - ran0.x);
      ran0.y = 0.5f * PI * ran0.y;
      ran1.x = 0.5f * PI * (1.0f - ran1.x);
      ran1.y = 0.5f * PI * ran1.y;

      data.V = angles_to_direction(ran0.x, ran0.y);

      BSDFSampleInfo info;
      bsdf_sample(data, make_ushort2(0, 0), info);

      sum_bounce += luminance(info.weight);

      vec3 L = angles_to_direction(ran1.x, ran1.y);

      bool is_transparent_pass;
      sum_light += luminance(bsdf_evaluate(data, L, BSDF_SAMPLING_GENERAL, is_transparent_pass, 2.0f * PI));
    }

    bounce[id] = sum_bounce / BRDF_UNITTEST_ITERATIONS;
    light[id]  = sum_light / BRDF_UNITTEST_ITERATIONS;

    id += blockDim.x * gridDim.x;
  }
}

extern "C" int device_brdf_unittest(const float tolerance) {
  DeviceBuffer* bounce = (DeviceBuffer*) 0;
  device_buffer_init(&bounce);
  device_buffer_malloc(bounce, sizeof(float), BRDF_UNITTEST_TOTAL);

  DeviceBuffer* light = (DeviceBuffer*) 0;
  device_buffer_init(&light);
  device_buffer_malloc(light, sizeof(float), BRDF_UNITTEST_TOTAL);

  brdf_unittest_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (float*) device_buffer_get_pointer(bounce), (float*) device_buffer_get_pointer(light));

  float* values_bounce = (float*) malloc(sizeof(float) * BRDF_UNITTEST_TOTAL);
  float* values_light  = (float*) malloc(sizeof(float) * BRDF_UNITTEST_TOTAL);

  device_buffer_download_full(bounce, (void*) values_bounce);
  device_buffer_download_full(light, (void*) values_light);

  int error = 0;

  printf("Bounce BRDF Unittest:\n");
  printf(" +-------------------------> Smoothness\n");

  for (int i = 0; i < BRDF_UNITTEST_STEPS_METALLIC; i++) {
    printf(" | ");
    if (i == 0 || i + 1 == BRDF_UNITTEST_STEPS_METALLIC) {
      for (int j = 0; j < BRDF_UNITTEST_STEPS_SMOOTHNESS; j++) {
        const float v = values_bounce[i * BRDF_UNITTEST_STEPS_SMOOTHNESS + j];
        if (v > 1.0001f || v < tolerance) {
          printf("\x1B[31m%.3f\x1B[0m ", v);
          error = 1;
        }
        else {
          printf("\x1B[32m%.3f\x1B[0m ", v);
        }
      }
    }
    else {
      for (int j = 0; j < BRDF_UNITTEST_STEPS_SMOOTHNESS; j++) {
        const float v = values_bounce[i * BRDF_UNITTEST_STEPS_SMOOTHNESS + j];
        if (v > 1.0001f || v < tolerance) {
          printf("\x1B[33m%.3f\x1B[0m ", v);
        }
        else {
          printf("\x1B[32m%.3f\x1B[0m ", v);
        }
      }
    }

    printf("\n");
  }

  printf(" V Metallic\n");
  printf("\n");
  printf("Light BRDF Unittest:\n");
  printf(" +-------------------------> Smoothness\n");

  for (int i = 0; i < BRDF_UNITTEST_STEPS_METALLIC; i++) {
    printf(" | ");
    if (i == 0 || i + 1 == BRDF_UNITTEST_STEPS_METALLIC) {
      for (int j = 0; j < BRDF_UNITTEST_STEPS_SMOOTHNESS; j++) {
        const float v = values_light[i * BRDF_UNITTEST_STEPS_SMOOTHNESS + j];
        if (v > 1.0001f || v < tolerance) {
          printf("\x1B[31m%.3f\x1B[0m ", v);
          error = 1;
        }
        else {
          printf("\x1B[32m%.3f\x1B[0m ", v);
        }
      }
    }
    else {
      for (int j = 0; j < BRDF_UNITTEST_STEPS_SMOOTHNESS; j++) {
        const float v = values_light[i * BRDF_UNITTEST_STEPS_SMOOTHNESS + j];
        if (v > 1.0001f || v < tolerance) {
          printf("\x1B[33m%.3f\x1B[0m ", v);
        }
        else {
          printf("\x1B[32m%.3f\x1B[0m ", v);
        }
      }
    }

    printf("\n");
  }

  printf(" V Metallic\n");
  printf("\n");

  free(values_bounce);
  free(values_light);
  device_buffer_destroy(&bounce);
  device_buffer_destroy(&light);

  return error;
}

#endif /* CU_BRDF_UNITTEST_H */
