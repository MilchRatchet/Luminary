#ifndef CU_BRDF_UNITTEST_H
#define CU_BRDF_UNITTEST_H

#include "brdf.cuh"
#include "buffer.h"
#include "utils.cuh"
#include "utils.h"

#define BRDF_UNITTEST_STEPS_SMOOTHNESS 20
#define BRDF_UNITTEST_STEPS_METALLIC 20
#define BRDF_UNITTEST_TOTAL (BRDF_UNITTEST_STEPS_SMOOTHNESS * BRDF_UNITTEST_STEPS_METALLIC)
#define BRDF_UNITTEST_ITERATIONS 1000000

__global__ void brdf_unittest_kernel(float* bounce, float* light) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int total = BRDF_UNITTEST_TOTAL;

  while (id < total) {
    const unsigned int x = id % BRDF_UNITTEST_STEPS_SMOOTHNESS;
    const unsigned int y = id / BRDF_UNITTEST_STEPS_SMOOTHNESS;

    const float smoothness = (1.0f / (BRDF_UNITTEST_STEPS_SMOOTHNESS - 1)) * x;
    const float metallic   = (1.0f / (BRDF_UNITTEST_STEPS_METALLIC - 1)) * y;

    float sum_bounce = 0.0f;
    float sum_light  = 0.0f;

    for (int i = 0; i < BRDF_UNITTEST_ITERATIONS; i++) {
      const float ran1 = 0.5f * PI * (1.0f - sqrtf(white_noise()));
      const float ran2 = 2.0f * PI * white_noise();
      const float ran3 = 0.5f * PI * (1.0f - sqrtf(white_noise()));
      const float ran4 = 2.0f * PI * white_noise();
      const vec3 V     = angles_to_direction(ran1, ran2);

      BRDFInstance brdf =
        brdf_get_instance(get_RGBAhalf(1.0f, 1.0f, 1.0f, 1.0f), V, get_vector(0.0f, 1.0f, 0.0f), 1.0f - smoothness, metallic);

      brdf = brdf_sample_ray(brdf, make_ushort2(0, 0), i);

      float weight = luminance(RGBAhalf_to_RGBF(brdf.term));

      sum_bounce += weight;

      brdf.L    = angles_to_direction(ran3, ran4);
      brdf.term = get_RGBAhalf(1.0f, 1.0f, 1.0f, 1.0f);

      brdf = brdf_evaluate(brdf);

      weight = luminance(RGBAhalf_to_RGBF(brdf.term));

      sum_light += weight;
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
