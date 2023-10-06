#ifndef CU_RANDOM_H
#define CU_RANDOM_H

#include "intrinsics.cuh"
#include "utils.cuh"

//
// This is an implementation of a very fast minimal state white noise generator. The goal was to minimize
// both register and memory usage. The original paper proposes a method of gathering keys for the
// generator but we use a simply key that is the number of alternating 0s and 1s added to the ID of the
// thread. This works well in practice and the both register and memory usage are tiny.
// The paper proposes a method of computing a 64bit output using 64bit inputs. Since we only have 32bit
// inputs, I adapted this method such that we obtain a 32bit output. For better performance, a version
// producing only 16 bits is used wherever we don't need a huge range of possible random numbers.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Wid20]
// Bernard Widynski, "Squares: A Fast Counter-Based RNG", 2020
// https://arxiv.org/abs/2004.06278

////////////////////////////////////////////////////////////////////
// Unsigned integer to float [0, 1] conversion
// Based on the uint64 conversion from Sebastiano Vigna and David Blackman (https://prng.di.unimi.it/)
////////////////////////////////////////////////////////////////////

__device__ float random_uint32_t_to_float(const uint32_t v) {
  const uint32_t i = 0x3F800000u | (v >> 9);

  return __uint_as_float(i) - 1.0f;
}

__device__ float random_uint16_t_to_float(const uint16_t v) {
  const uint32_t i = 0x3F800000u | (((uint32_t) v) << 7);

  return __uint_as_float(i) - 1.0f;
}

////////////////////////////////////////////////////////////////////
// Random number generators
////////////////////////////////////////////////////////////////////

// This is the base generator for random 32 bits.
__device__ uint32_t random_uint32_t_base(const uint32_t key_offset, const uint32_t offset) {
  // Key is supposed to be a number with roughly the same amount of 0 bits as 1 bits.
  // This key here seems to work well.
  const uint32_t key     = key_offset + 0x55555555;
  const uint32_t counter = offset;

  uint32_t x = counter * key;
  uint32_t y = counter * key;
  uint32_t z = y + key;

  x = x * x + y;
  x = __uswap16p(x);

  x = x * x + z;
  x = __uswap16p(x);

  x = x * x + y;
  x = __uswap16p(x);

  x = x * x + z;
  z = x;
  x = __uswap16p(x);

  return z ^ ((x * x + y) >> 16);
}

// This is the base generator for random 16 bits.
__device__ uint16_t random_uint16_t_base(const uint32_t key_offset, const uint32_t offset) {
  // Key is supposed to be a number with roughly the same amount of 0 bits as 1 bits.
  // This key here seems to work well.
  const uint32_t key     = key_offset + 0x55555555;
  const uint32_t counter = offset;

  uint32_t x = counter * key;
  uint32_t y = counter * key;
  uint32_t z = y + key;

  x = x * x + y;
  x = __uswap16p(x);

  x = x * x + z;
  x = __uswap16p(x);

  return (x * x + y) >> 16;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ uint32_t random_uint32_t(const uint32_t offset) {
  return random_uint32_t_base(THREAD_ID, offset);
}

__device__ uint16_t random_uint16_t(const uint32_t offset) {
  return random_uint16_t_base(THREAD_ID, offset);
}

__device__ float white_noise_precise_offset(const uint32_t offset) {
  return random_uint32_t_to_float(random_uint32_t(offset));
}

__device__ float white_noise_offset(const uint32_t offset) {
  return random_uint16_t_to_float(random_uint16_t(offset));
}

__device__ float white_noise_precise() {
  return white_noise_precise_offset(device.ptrs.randoms[THREAD_ID]++);
}

__device__ float white_noise() {
  return white_noise_offset(device.ptrs.randoms[THREAD_ID]++);
}

////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////

#ifndef RANDOM_NO_KERNELS
__global__ void initialize_randoms() {
  device.ptrs.randoms[THREAD_ID] = 1;
}
#endif

#endif /* CU_RANDOM_H */
