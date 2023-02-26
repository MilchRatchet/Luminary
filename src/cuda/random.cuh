#ifndef CU_RANDOM_H
#define CU_RANDOM_H

#include "utils.cuh"

//
// This is an implementation of a very fast minimal state white noise generator. The goal was to minimize
// both register and memory usage. The original paper proposes a method of gathering keys for the
// generator but we use a simply key that is the number of alternating 0s and 1s added to the ID of the
// thread. This works well in practice and the both register and memory usage are tiny.
// The paper proposes a method of computing a 64bit output using 64bit inputs. Since we only have 32bit
// inputs, I adapted this method such that we obtain a 32bit output.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Wid20]
// Bernard Widynski, "Squares: A Fast Counter-Based RNG", 2020
// https://arxiv.org/abs/2004.06278

__device__ float white_noise_offset(const uint32_t offset) {
  // Key is supposed to be a number with roughly the same amount of 0 bits as 1 bits.
  // This key here seems to work well.
  const uint32_t key     = threadIdx.x + blockIdx.x * blockDim.x + 0x55555555;
  const uint32_t counter = offset;

  uint32_t x = counter * key;
  uint32_t y = counter * key;
  uint32_t z = y + key;

  x = x * x + y;
  x = (x >> 16) | (x << 16);

  x = x * x + z;
  x = (x >> 16) | (x << 16);

  x = x * x + y;
  x = (x >> 16) | (x << 16);

  x = x * x + z;
  z = x;
  x = (x >> 16) | (x << 16);

  x = z ^ ((x * x + y) >> 16);

  return ((float) x) / ((float) UINT32_MAX);
}

__device__ float white_noise() {
  return white_noise_offset(device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x]++);
}

__global__ void initialize_randoms() {
  device.ptrs.randoms[threadIdx.x + blockIdx.x * blockDim.x] = 1;
}

#endif /* CU_RANDOM_H */
