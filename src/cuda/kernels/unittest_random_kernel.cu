#include "random.cuh"
#include "utils.cuh"

#define DEPTH_PER_ITERATION 10

__global__ void unittest_random_kernel(float* results, uint32_t total, uint32_t iterations) {
  unsigned int id = THREAD_ID;

  while (id < total) {
    float depth_sums[64];

    for (int d = 0; d < 64; d++) {
      float sum = 0.0f;
      for (int i = 0; i < iterations; i++) {
        sum += quasirandom_sequence_1D_base(id, 0, i, d);
      }
      depth_sums[d] = sum / iterations;
    }

    float sum = 0.0f;

    for (int d = 0; d < 64; d++) {
      sum += depth_sums[d];
    }

    results[id] = sum / 64;

    id += blockDim.x * gridDim.x;
  }
}
