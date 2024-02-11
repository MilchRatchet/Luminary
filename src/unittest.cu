#include <stdio.h>

#include "buffer.h"
#include "cuda/kernels_unittest.cuh"
#include "log.h"
#include "utils.cuh"
#include "utils.h"

#define BRDF_UNITTEST_STEPS 20
#define BRDF_UNITTEST_TOTAL (BRDF_UNITTEST_STEPS * BRDF_UNITTEST_STEPS)
#define BRDF_UNITTEST_ITERATIONS 1000000

#define RANDOM_UNITTEST_ITERATIONS 25000
#define RANDOM_UNITTEST_TOTAL 500

extern "C" int unittest_random() {
  DeviceBuffer* results = (DeviceBuffer*) 0;
  device_buffer_init(&results);
  device_buffer_malloc(results, sizeof(float), RANDOM_UNITTEST_TOTAL);

  unittest_random_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (float*) device_buffer_get_pointer(results), RANDOM_UNITTEST_TOTAL, RANDOM_UNITTEST_ITERATIONS);

  float* values_results = (float*) malloc(sizeof(float) * RANDOM_UNITTEST_TOTAL);

  device_buffer_download_full(results, (void*) values_results);

  int error = 0;

  printf("Random Unittest:\n");
  for (int i = 0; i < RANDOM_UNITTEST_TOTAL; i++) {
    if (fabsf(0.5f - values_results[i]) > 1e-3f) {
      error = 1;
      printf("%u: \x1B[31m%f\x1B[0m\n", i, values_results[i]);
    }
    else {
      printf("%u: \x1B[32m%f\x1B[0m\n", i, values_results[i]);
    }
  }

  free(values_results);
  device_buffer_destroy(&results);

  return error;
}

extern "C" int unittest_brdf(const float tolerance) {
  DeviceBuffer* bounce = (DeviceBuffer*) 0;
  device_buffer_init(&bounce);
  device_buffer_malloc(bounce, sizeof(float), BRDF_UNITTEST_TOTAL);

  DeviceBuffer* light = (DeviceBuffer*) 0;
  device_buffer_init(&light);
  device_buffer_malloc(light, sizeof(float), BRDF_UNITTEST_TOTAL);

  unittest_brdf_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (float*) device_buffer_get_pointer(bounce), (float*) device_buffer_get_pointer(light), BRDF_UNITTEST_TOTAL, BRDF_UNITTEST_STEPS,
    BRDF_UNITTEST_ITERATIONS);

  float* values_bounce = (float*) malloc(sizeof(float) * BRDF_UNITTEST_TOTAL);
  float* values_light  = (float*) malloc(sizeof(float) * BRDF_UNITTEST_TOTAL);

  device_buffer_download_full(bounce, (void*) values_bounce);
  device_buffer_download_full(light, (void*) values_light);

  int error = 0;

  printf("Bounce BRDF Unittest:\n");
  printf(" +-------------------------> Smoothness\n");

  for (int i = 0; i < BRDF_UNITTEST_STEPS; i++) {
    printf(" | ");
    if (i == 0 || i + 1 == BRDF_UNITTEST_STEPS) {
      for (int j = 0; j < BRDF_UNITTEST_STEPS; j++) {
        const float v = values_bounce[i * BRDF_UNITTEST_STEPS + j];
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
      for (int j = 0; j < BRDF_UNITTEST_STEPS; j++) {
        const float v = values_bounce[i * BRDF_UNITTEST_STEPS + j];
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

  for (int i = 0; i < BRDF_UNITTEST_STEPS; i++) {
    printf(" | ");
    if (i == 0 || i + 1 == BRDF_UNITTEST_STEPS) {
      for (int j = 0; j < BRDF_UNITTEST_STEPS; j++) {
        const float v = values_light[i * BRDF_UNITTEST_STEPS + j];
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
      for (int j = 0; j < BRDF_UNITTEST_STEPS; j++) {
        const float v = values_light[i * BRDF_UNITTEST_STEPS + j];
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
