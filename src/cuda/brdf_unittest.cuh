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
      const float ran1 = 0.5f * PI * sqrtf(white_noise());
      const float ran2 = 2.0f * PI * white_noise();
      const float ran3 = 0.5f * PI * sqrtf(white_noise());
      const float ran4 = 2.0f * PI * white_noise();
      const vec3 V     = sample_ray_from_angles_and_vector(ran1, ran2, get_vector(1.0f, 0.0f, 0.0f));
      vec3 L;

      RGBAhalf brdf = get_RGBAhalf(1.0f, 1.0f, 1.0f, 1.0f);

      brdf_sample_ray(
        L, brdf, make_ushort2(0, 0), i, get_color(1.0f, 1.0f, 1.0f), V, get_vector(1.0f, 0.0f, 0.0f), get_vector(1.0f, 0.0f, 0.0f),
        1.0f - smoothness, metallic);

      float weight = luminance(RGBAhalf_to_RGBF(brdf));

      sum_bounce += weight;

      L = sample_ray_from_angles_and_vector(ran3, ran4, get_vector(1.0f, 0.0f, 0.0f));

      brdf = brdf_evaluate(get_color(1.0f, 1.0f, 1.0f), V, L, get_vector(1.0f, 0.0f, 0.0f), 1.0f - smoothness, metallic);

      weight = 2.0f * PI * luminance(RGBAhalf_to_RGBF(brdf));

      sum_light += weight;
    }

    bounce[id] = sum_bounce / BRDF_UNITTEST_ITERATIONS;
    light[id]  = sum_light / BRDF_UNITTEST_ITERATIONS;

    id += blockDim.x * gridDim.x;
  }
}

extern "C" int brdf_unittest(const float tolerance) {
  DeviceBuffer* bounce;
  device_buffer_init(&bounce);
  device_buffer_malloc(bounce, sizeof(float), BRDF_UNITTEST_TOTAL);

  DeviceBuffer* light;
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

    printf("\n");
  }

  printf(" V Metallic\n");
  printf("\n");
  printf("Light BRDF Unittest:\n");
  printf(" +-------------------------> Smoothness\n");

  for (int i = 0; i < BRDF_UNITTEST_STEPS_METALLIC; i++) {
    printf(" | ");
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

    printf("\n");
  }

  printf(" V Metallic\n");
  printf("\n");

  free(values_bounce);
  free(values_light);
  device_buffer_destroy(bounce);
  device_buffer_destroy(light);

  return error;
}

#endif /* CU_BRDF_UNITTEST_H */
