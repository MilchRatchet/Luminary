#ifndef CU_RANDOM_UNITTEST_H
#define CU_RANDOM_UNITTEST_H

#include "buffer.h"
#include "random.cuh"
#include "utils.cuh"
#include "utils.h"

#define RANDOM_UNITTEST_ITERATIONS 25000
#define RANDOM_UNITTEST_DEPTH 64
#define RANDOM_UNITTEST_TOTAL 500

__global__ void random_unittest_kernel(float* results) {
  unsigned int id = THREAD_ID;

  const unsigned int total = RANDOM_UNITTEST_TOTAL;

  while (id < total) {
    float depth_sums[RANDOM_UNITTEST_DEPTH];

    for (int d = 0; d < RANDOM_UNITTEST_DEPTH; d++) {
      float sum = 0.0f;
      for (int i = 0; i < RANDOM_UNITTEST_ITERATIONS; i++) {
        sum += quasirandom_sequence_1D_base(id, make_ushort2(0, 0), i, d);
      }
      depth_sums[d] = sum / RANDOM_UNITTEST_ITERATIONS;
    }

    float sum = 0.0f;

    for (int d = 0; d < RANDOM_UNITTEST_DEPTH; d++) {
      sum += depth_sums[d];
    }

    results[id] = sum / RANDOM_UNITTEST_DEPTH;

    id += blockDim.x * gridDim.x;
  }
}

extern "C" int device_random_unittest() {
  DeviceBuffer* results = (DeviceBuffer*) 0;
  device_buffer_init(&results);
  device_buffer_malloc(results, sizeof(float), RANDOM_UNITTEST_TOTAL);

  random_unittest_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((float*) device_buffer_get_pointer(results));

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

#endif /* CU_RANDOM_UNITTEST_H */
