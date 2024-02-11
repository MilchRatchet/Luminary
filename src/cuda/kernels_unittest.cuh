#ifndef CU_KERNELS_UNITTEST_H
#define CU_KERNELS_UNITTEST_H

#include "utils.cuh"

__global__ void unittest_brdf_kernel(float* bounce, float* light, uint32_t total, uint32_t steps, uint32_t iterations);
__global__ void unittest_random_kernel(float* results, uint32_t total, uint32_t iterations);

#endif /* CU_KERNELS_UNITTEST_H */
