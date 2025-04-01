#ifndef CU_RANDOM_H
#define CU_RANDOM_H

#include "intrinsics.cuh"
#include "utils.cuh"

#define BLUENOISE_TEX_DIM 256
#define BLUENOISE_TEX_DIM_MASK 0xFF

enum QuasiRandomTargetAllocation : uint32_t {
  QUASI_RANDOM_TARGET_ALLOCATION_LENS                     = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_MICROFACET          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_DIFFUSE             = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_REFRACTION          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_DIR_CHOICE        = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_DIR               = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_OPACITY           = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LENS_BLADE               = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_VOLUME_INTERSECTION      = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_RUSSIAN_ROULETTE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAMERA_JITTER            = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAMERA_TIME              = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_STEP_OFFSET        = 3,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_STEP_COUNT         = 3,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_DIR                = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_SKY_STEP_OFFSET          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_SKY_INSCATTERING_STEP    = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_RIS_DIFFUSE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_RIS_REFRACTION      = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_LIGHT_ID             = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_RAY_DIR              = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_RESAMPLING           = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_LIGHT_TREE           = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_INITIAL          = 128,
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_RESAMPLE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_SUN_DIR          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_BSDF           = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_BSDF               = 2,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_RAY            = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_RIS_RESAMPLING = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_PREFIX            = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_LIGHT_TREE        = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_LIGHT_POINT       = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_DISTANCE          = (32 * 8),
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_PHASE             = (32 * 8),
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_VERTEX_COUNT      = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_RESAMPLING        = 1
} typedef QuasiRandomTargetAllocation;

#define QUASI_RANDOM_TARGET_ALLOC(name) (QUASI_RANDOM_TARGET_##name + QUASI_RANDOM_TARGET_ALLOCATION_##name)

enum QuasiRandomTarget : uint32_t {
  QUASI_RANDOM_TARGET_LENS                     = 0,
  QUASI_RANDOM_TARGET_BSDF_MICROFACET          = QUASI_RANDOM_TARGET_ALLOC(LENS),
  QUASI_RANDOM_TARGET_BSDF_DIFFUSE             = QUASI_RANDOM_TARGET_ALLOC(BSDF_MICROFACET),
  QUASI_RANDOM_TARGET_BSDF_REFRACTION          = QUASI_RANDOM_TARGET_ALLOC(BSDF_DIFFUSE),
  QUASI_RANDOM_TARGET_BOUNCE_OPACITY           = QUASI_RANDOM_TARGET_ALLOC(BSDF_REFRACTION),
  QUASI_RANDOM_TARGET_LENS_BLADE               = QUASI_RANDOM_TARGET_ALLOC(BOUNCE_OPACITY),
  QUASI_RANDOM_TARGET_VOLUME_INTERSECTION      = QUASI_RANDOM_TARGET_ALLOC(LENS_BLADE),
  QUASI_RANDOM_TARGET_RUSSIAN_ROULETTE         = QUASI_RANDOM_TARGET_ALLOC(VOLUME_INTERSECTION),
  QUASI_RANDOM_TARGET_CAMERA_JITTER            = QUASI_RANDOM_TARGET_ALLOC(RUSSIAN_ROULETTE),
  QUASI_RANDOM_TARGET_CAMERA_TIME              = QUASI_RANDOM_TARGET_ALLOC(CAMERA_JITTER),
  QUASI_RANDOM_TARGET_CLOUD_STEP_OFFSET        = QUASI_RANDOM_TARGET_ALLOC(CAMERA_TIME),
  QUASI_RANDOM_TARGET_CLOUD_STEP_COUNT         = QUASI_RANDOM_TARGET_ALLOC(CLOUD_STEP_OFFSET),
  QUASI_RANDOM_TARGET_CLOUD_DIR                = QUASI_RANDOM_TARGET_ALLOC(CLOUD_STEP_COUNT),
  QUASI_RANDOM_TARGET_SKY_STEP_OFFSET          = QUASI_RANDOM_TARGET_ALLOC(CLOUD_DIR),
  QUASI_RANDOM_TARGET_SKY_INSCATTERING_STEP    = QUASI_RANDOM_TARGET_ALLOC(SKY_STEP_OFFSET),
  QUASI_RANDOM_TARGET_BSDF_RIS_DIFFUSE         = QUASI_RANDOM_TARGET_ALLOC(SKY_INSCATTERING_STEP),
  QUASI_RANDOM_TARGET_BSDF_RIS_REFRACTION      = QUASI_RANDOM_TARGET_ALLOC(BSDF_RIS_DIFFUSE),
  QUASI_RANDOM_TARGET_RIS_LIGHT_ID             = QUASI_RANDOM_TARGET_ALLOC(BSDF_RIS_REFRACTION),
  QUASI_RANDOM_TARGET_RIS_RAY_DIR              = QUASI_RANDOM_TARGET_ALLOC(RIS_LIGHT_ID),
  QUASI_RANDOM_TARGET_RIS_RESAMPLING           = QUASI_RANDOM_TARGET_ALLOC(RIS_RAY_DIR),
  QUASI_RANDOM_TARGET_RIS_LIGHT_TREE           = QUASI_RANDOM_TARGET_ALLOC(RIS_RESAMPLING),
  QUASI_RANDOM_TARGET_CAUSTIC_INITIAL          = QUASI_RANDOM_TARGET_ALLOC(RIS_LIGHT_TREE),
  QUASI_RANDOM_TARGET_CAUSTIC_RESAMPLE         = QUASI_RANDOM_TARGET_ALLOC(CAUSTIC_INITIAL),
  QUASI_RANDOM_TARGET_CAUSTIC_SUN_DIR          = QUASI_RANDOM_TARGET_ALLOC(CAUSTIC_RESAMPLE),
  QUASI_RANDOM_TARGET_LIGHT_SUN_BSDF           = QUASI_RANDOM_TARGET_ALLOC(CAUSTIC_SUN_DIR),
  QUASI_RANDOM_TARGET_LIGHT_BSDF               = QUASI_RANDOM_TARGET_ALLOC(LIGHT_SUN_BSDF),
  QUASI_RANDOM_TARGET_LIGHT_SUN_RAY            = QUASI_RANDOM_TARGET_ALLOC(LIGHT_BSDF),
  QUASI_RANDOM_TARGET_LIGHT_SUN_RIS_RESAMPLING = QUASI_RANDOM_TARGET_ALLOC(LIGHT_SUN_RAY),
  QUASI_RANDOM_TARGET_BRIDGE_PREFIX            = QUASI_RANDOM_TARGET_ALLOC(LIGHT_SUN_RIS_RESAMPLING),
  QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE        = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_PREFIX),
  QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT       = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_LIGHT_TREE),
  QUASI_RANDOM_TARGET_BRIDGE_DISTANCE          = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_LIGHT_POINT),
  QUASI_RANDOM_TARGET_BRIDGE_PHASE             = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_DISTANCE),
  QUASI_RANDOM_TARGET_BRIDGE_VERTEX_COUNT      = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_PHASE),
  QUASI_RANDOM_TARGET_BRIDGE_RESAMPLING        = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_VERTEX_COUNT),

  QUASI_RANDOM_TARGET_COUNT = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_RESAMPLING)
} typedef QuasiRandomTarget;

// Target reuse, these targets are used mutually exclusive with the other
#define QUASI_RANDOM_TARGET_BSDF_VOLUME QUASI_RANDOM_TARGET_BSDF_MICROFACET
#define QUASI_RANDOM_TARGET_BSDF_VOLUME_CHOISE QUASI_RANDOM_TARGET_BSDF_DIFFUSE

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Wid20]
// Bernard Widynski, "Squares: A Fast Counter-Based RNG", 2020
// https://arxiv.org/abs/2004.06278

// [Rob18]
// Martin Roberts, "The Unreasonable Effectiveness of Quasirandom Sequences", 2018
// https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

// [Wol22]
// A. Wolfe, N. Morrical, T. Akenine-Möller and R. Ramamoorthi, "Spatiotemporal Blue Noise Masks"
// Eurographics Symposium on Rendering, pp. 117-126, 2022.

// [Bel21]
// Laurent Belcour and Eric Heitz, "Lessons Learned and Improvements when Building Screen-Space Samplers with Blue-Noise Error Distribution"
// ACM SIGGRAPH 2021 Talks, pp. 1-2, 2021.

// [Bur20]
// Brent Burley, "Practical Hash-based Owen Scrambling"
// Journal of Computer Graphics Techniques (JCGT), pp. 1-20, 2020.

// [Ahm24]
// Abdalla G. M. Ahmed, "An Implementation Algorithm of 2D Sobol Sequence Fast, Elegant, and Compact"
// Eurographics Symposium on Rendering, 2024.

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
// Utils
////////////////////////////////////////////////////////////////////

__device__ float random_saturate(const float random) {
  return fminf(fmaxf(random, 0.0f), 1.0f - 8.0f * eps);
}

////////////////////////////////////////////////////////////////////
// Random number generators
////////////////////////////////////////////////////////////////////

// This is the base generator for random 32 bits.
__device__ uint32_t random_uint32_t_base(const uint32_t key_offset, const uint32_t offset) {
  const uint32_t key     = key_offset;
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

  return z ^ (x * x + y);
}

// This is the base generator for random 16 bits.
__device__ uint16_t random_uint16_t_base(const uint32_t key_offset, const uint32_t offset) {
  const uint32_t key     = key_offset;
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
// Kronecker [Rob18]
////////////////////////////////////////////////////////////////////

// Integer fractions of the actual numbers
#define R1_PHI1 2654435769u /* 0.61803398875f */
#define R2_PHI1 3242174889u /* 0.7548776662f  */
#define R2_PHI2 2447445413u /* 0.56984029f    */

__device__ uint32_t random_r1(const uint32_t offset) {
  return (offset + 1) * R1_PHI1;
}

__device__ uint2 random_r2(const uint32_t index) {
  const uint32_t v1 = (1 + index) * R2_PHI1;
  const uint32_t v2 = (1 + index) * R2_PHI2;

  return make_uint2(v1, v2);
}

////////////////////////////////////////////////////////////////////
// Sobol [Bur20] [Ahm24]
////////////////////////////////////////////////////////////////////

__device__ uint32_t random_laine_karras_permutation(uint32_t x, uint32_t seed) {
  x += seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}

__device__ uint32_t random_nested_uniform_scramble_base2(uint32_t x, uint32_t seed) {
  x = __brev(x);
  x = random_laine_karras_permutation(x, seed);
  x = __brev(x);
  return x;
}

__device__ uint32_t random_hash_combine(uint32_t seed, uint32_t v) {
  return seed ^ (v + (seed << 6) + (seed >> 2));
}

__device__ uint32_t random_sobol_P(uint32_t v) {
  v ^= v << 16;
  v ^= (v & 0x00FF00FF) << 8;
  v ^= (v & 0x0F0F0F0F) << 4;
  v ^= (v & 0x33333333) << 2;
  v ^= (v & 0x55555555) << 1;
  return v;
}

__device__ uint2 random_sobol_base(uint32_t offset, const uint32_t seed) {
  // The index into the sequence is given by:
  // const uint32_t index = random_nested_uniform_scramble_base2(offset, seed);
  // Then applying the J matrix to the index as in [Ahm24]:
  // const uint32_t J = __brev(index);
  // We can get rid of the 2 consecutive __brev here and obtain J as follows:
  const uint32_t J = random_laine_karras_permutation(__brev(offset), seed);

  return make_uint2(J, random_sobol_P(J));
}

__device__ uint2 random_sobol(const uint32_t offset, const uint32_t dimension) {
  const uint32_t seed = random_uint32_t_base(0xfcbd6e15, dimension);

  const uint2 sobol = random_sobol_base(offset, seed);

  const uint32_t v1 = random_nested_uniform_scramble_base2(sobol.x, random_hash_combine(seed, 0));
  const uint32_t v2 = random_nested_uniform_scramble_base2(sobol.y, random_hash_combine(seed, 1));

  return make_uint2(v1, v2);
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

__device__ uint32_t random_uint32_t(const uint32_t offset) {
  return random_uint32_t_base(0xfcbd6e15, offset);
}

__device__ uint16_t random_uint16_t(const uint32_t offset) {
  return random_uint16_t_base(0xfcbd6e15, offset);
}

__device__ float white_noise_precise_offset(const uint32_t offset) {
  return random_uint32_t_to_float(random_uint32_t(offset));
}

__device__ float white_noise_offset(const uint32_t offset) {
  return random_uint16_t_to_float(random_uint16_t(offset));
}

__device__ uint2 random_blue_noise_mask_2D(const uint32_t x, const uint32_t y) {
  const uint32_t pixel      = (x & BLUENOISE_TEX_DIM_MASK) + (y & BLUENOISE_TEX_DIM_MASK) * BLUENOISE_TEX_DIM;
  const uint32_t blue_noise = __ldg(device.ptrs.bluenoise_2D + pixel);

  return make_uint2(blue_noise & 0xFFFF0000, blue_noise << 16);
}

////////////////////////////////////////////////////////////////////
// Warning: The lowest bit is always 0 for these random numbers.
////////////////////////////////////////////////////////////////////

__device__ uint2
  quasirandom_sequence_2D_base(const uint32_t target, const ushort2 pixel, const uint32_t sequence_id, const uint32_t depth) {
  uint32_t dimension_index = target + depth * QUASI_RANDOM_TARGET_COUNT;

  uint2 quasi = random_sobol(sequence_id, dimension_index);

  const uint2 pixel_offset = random_r2(dimension_index);
  const uint2 blue_noise   = random_blue_noise_mask_2D(pixel.x + (pixel_offset.x >> 24), pixel.y + (pixel_offset.y >> 24));

  quasi.x += blue_noise.x;
  quasi.y += blue_noise.y;

  return quasi;
}

__device__ float2
  quasirandom_sequence_2D_base_float(const uint32_t target, const ushort2 pixel, const uint32_t sequence_id, const uint32_t depth) {
  const uint2 quasi = quasirandom_sequence_2D_base(target, pixel, sequence_id, depth);

  return make_float2(random_uint32_t_to_float(quasi.x), random_uint32_t_to_float(quasi.y));
}

__device__ uint32_t
  quasirandom_sequence_1D_base(const uint32_t target, const ushort2 pixel, const uint32_t sequence_id, const uint32_t depth) {
  return quasirandom_sequence_2D_base(target, pixel, sequence_id, depth).x;
}

__device__ float quasirandom_sequence_1D_base_float(
  const uint32_t target, const ushort2 pixel, const uint32_t sequence_id, const uint32_t depth) {
  return random_uint32_t_to_float(quasirandom_sequence_1D_base(target, pixel, sequence_id, depth));
}

__device__ float quasirandom_sequence_1D(const uint32_t target, const ushort2 pixel) {
  return quasirandom_sequence_1D_base_float(target, pixel, device.state.sample_id, device.state.depth);
}

// This is a global version that is constant within a given frame.
__device__ float quasirandom_sequence_1D_global(const uint32_t target) {
  return quasirandom_sequence_1D_base_float(target, make_ushort2(0, 0), device.state.sample_id, 0);
}

__device__ float2 quasirandom_sequence_2D(const uint32_t target, const ushort2 pixel) {
  return quasirandom_sequence_2D_base_float(target, pixel, device.state.sample_id, device.state.depth);
}

// This is a global version that is constant within a given frame.
__device__ float2 quasirandom_sequence_2D_global(const uint32_t target) {
  return quasirandom_sequence_2D_base_float(target, make_ushort2(0, 0), device.state.sample_id, 0);
}

__device__ float random_dither_mask(const uint32_t x, const uint32_t y) {
  const uint32_t pixel      = (x & BLUENOISE_TEX_DIM_MASK) + (y & BLUENOISE_TEX_DIM_MASK) * BLUENOISE_TEX_DIM;
  const uint16_t blue_noise = __ldg(device.ptrs.bluenoise_1D + pixel);

  return random_uint16_t_to_float(blue_noise);
}

__device__ float random_grain_mask(const uint32_t x, const uint32_t y) {
  return white_noise_offset(x + y * device.settings.width);
}

#endif /* CU_RANDOM_H */
