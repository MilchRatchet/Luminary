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

enum QuasiRandomTarget : uint32_t {
  QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE           = 0,   /* 1 */
  QUASI_RANDOM_TARGET_BOUNCE_DIR                  = 1,   /* 2 */
  QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY         = 3,   /* 1 */
  QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY          = 4,   /* 1 */
  QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY_ROULETTE = 5,   /* 1 */
  QUASI_RANDOM_TARGET_RESTIR_CHOICE               = 6,   /* 2 + 128 */
  QUASI_RANDOM_TARGET_RESTIR_GENERATION           = 136, /* 128 */
  QUASI_RANDOM_TARGET_LENS                        = 264, /* 2 */
  QUASI_RANDOM_TARGET_VOLUME_DIST                 = 266, /* 1 */
  QUASI_RANDOM_TARGET_RUSSIAN_ROULETTE            = 267  /* 1 */
} typedef QuasiRandomTarget;

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Wid20]
// Bernard Widynski, "Squares: A Fast Counter-Based RNG", 2020
// https://arxiv.org/abs/2004.06278

// [Rob18]
// Martin Roberts, "The Unreasonable Effectiveness of Quasirandom Sequences", 2018
// https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

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

// Integer fractions of the actual numbers
#define R1_PHI 2654435768u  /*0.61803398875f*/
#define R2_PHI1 3242174888u /*0.7548776662f*/
#define R2_PHI2 2447445413u /*0.56984029f*/

__device__ float random_r1(const uint32_t offset) {
  const uint32_t v = offset * R1_PHI;

  return random_uint32_t_to_float(v);
}

__device__ float2 random_r2(const uint32_t offset) {
  const uint32_t v1 = offset * R2_PHI1;
  const uint32_t v2 = offset * R2_PHI2;

  return make_float2(random_uint32_t_to_float(v1), random_uint32_t_to_float(v2));
}

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

__device__ float quasirandom_sequence_1D(const uint32_t target, const uint32_t pixel) {
  const float offset = random_uint16_t_to_float(random_uint16_t_base(pixel + (device.depth << 24), target));

  float quasi = random_r1(device.temporal_frames);

  quasi += offset;
  quasi -= truncf(quasi);

  return quasi;
}

__device__ float2 quasirandom_sequence_2D(const uint32_t target, const uint32_t pixel) {
  const float offset1 = random_uint16_t_to_float(random_uint16_t_base(pixel + (device.depth << 24), target));
  const float offset2 = random_uint16_t_to_float(random_uint16_t_base(pixel + (device.depth << 24), target + 1));

  float2 quasi = random_r2(device.temporal_frames);

  quasi.x += offset1;
  quasi.x -= truncf(quasi.x);

  quasi.y += offset2;
  quasi.y -= truncf(quasi.y);

  return quasi;
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
