#ifndef CU_RANDOM_H
#define CU_RANDOM_H

#include <curand_kernel.h>

#include "utils.cuh"

__device__ float white_noise() {
  return curand_uniform(((curandStateXORWOW_t*) device.ptrs.randoms) + threadIdx.x + blockIdx.x * blockDim.x);
}

__global__ void initialize_randoms() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  curandStateXORWOW_t state;
  curand_init(id, 0, 0, &state);
  ((curandStateXORWOW_t*) device.ptrs.randoms)[id] = state;
}

#endif /* CU_RANDOM_H */
