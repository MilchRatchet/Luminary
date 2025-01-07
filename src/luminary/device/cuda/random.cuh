#ifndef CU_RANDOM_H
#define CU_RANDOM_H

#include "intrinsics.cuh"
#include "utils.cuh"

//
// This is the implementation for the random numbers used in Luminary.
// First and foremost what we are trying to do is solve an integral using the Monte Carlo method.
// This integral is the rendering equation and we need to solve it for each pixel.
// A classic approach to accelerating Monte Carlo methods is using low-discrepancy random numbers.
// This results in the so called quasi-Monte Carlo method. We can summarize the idea of that as follows:
// Uncorrelated random numbers can clump, that is, multiple random numbers in a row could end up being
// very close to each other. If we now think about integrating a function then another classical approach is
// evaluating the function at equidistant points. For that we see that the equidistant points manage to cover
// a large range of the functions domain and will only fail if the function has very different outputs
// inbetween those points. Our clumped up random points however will fail to give a good representation of our
// function even if the function is well behaved since we are not effectively covering the functions domain.
// Hence we use low-discrepancy random numbers which are uniformly distributed but neighbouring numbers
// in the sequence are far apart from another. The analogue to the equidistant points is then a sequence
// of points that each split the existing sampling gaps in half, i.e., 1/2, 1/4, 3/4, 1/8, 5/8 and so on.
// This sequence is called the "van der Corput" sequence. While the points are low-discrepency, they
// have a visible pattern and hence more "randomly" distributed points are to be preferred.
// A multitude of random number sequences have been proposed, one of the most notable is the Halton and
// the Sobol sequence. Especially the latter in combination with Owen scrambling is known to produce
// high quality low-discrepency numbers without any visible pattern. Another very simple sequence
// is the sequence in which the i-th number is given by i * \alpha (mod 1). Here \alpha can be
// chosen to be any irrational number. In particular, choosing \alpha to be the inverse of the
// golden ratio turns out to give the sequence of lowest possible discrepancy (although via
// Mobius transformation we can obtain similarly effective values of \alpha) [Rob18].
//
// So we have seen multiple ways of obtaining low-discrepancy random numbers and why they are useful.
// While all of these can be generalized to give D-dimensional random numbers that are low-discrepancy
// in the D-dimensional metric space. The number D of random numbers that we need to render a single sample
// per pixel is a) often way too large to compute such a D-dimensional low-discrepancy random number and
// b) it is often not known a-priori how many random numbers we even need. Hence the idea is to generate
// a D-dimensional random number by computing one random number and modifying it using a different sequence
// of random numbers. All entries inside this D-dimensional random number must be uncorrelated or else we would
// obtain a biased result. You can think of this as we want the decision to use a diffuse importance sampled
// direction to not be correlated with the actual sampled direction obtained from that method, i.e., we don't
// want all diffuse samples to go up. Hence we modify them using a uncorrelated sequence of random numbers.
// For this we take a direction vector s of dimension D and obtain our low-discrepancy D-dimensional random numbers
// each by multiplying s with the low-discrepancy number and then applying modulo 1 to wrap the entries back into
// the [0,1) range.
//
// Now that we can generate low-discrepancy D-dimensional random numbers to perform quasi-Monte Carlo integration for
// each pixel, we need to think about the sequences for all our pixels. If we were to use the same random number
// sequence for each pixel, we would obtain almost identical errors in each pixel. On the surface this does not sound
// too bad but visually this means that the error is much more prominent and the image after a few samples per pixel
// will not look anything like the final converged image. To fix this we can apply a shift of the form a + b (mod 1)
// to the D-dimensional low-discrepancy random numbers that is different for each pixel. This is also called a Cranley-Patterson rotation.
// This will distribute the error across pixels which gives a much more visually pleasing result. In fact, this
// is the point where our images start to become noisy, without this they would look almost noise free but heavily biased
// instead. It is also much clearer to understand if an image is (pretty much) converged based on the noise levels
// as otherwise it is impossible to visually identify the error if no reference image exists. For the choice of the shift number
// we could use standard uncorrelated random numbers but if we do that we would run into clumping again which means
// that the error would not be evenly distributed over the image. Hence we again use low-discrepancy numbers to perform
// this shift. A common approach for this are so called blue noise masks which are specifically crafted images that shift
// each sequence of D-dimensional random numbers such that the error is very evenly distributed across pixels. Since these
// images are only of low dimensionality, we use low-discrepancy numbers to shift our pixel coordinates such that we effectively
// obtain D many images. Alternatively, other approach likes using Hilbert or Morton indexing together with scrambling have been
// shown to be a good alternative that don't need any precomputed texture. They work by having a very large base sequence of
// low-discrepancy numbers and computing a different index in this sequence for each pixel.
//
// Now we come to the actual implementation here. First it is important to note that this whole area of low-discrepancy
// random numbers for Monte Carlo rendering is a very ongoing research topic currently and it is not always clear what
// is the best approach now and in the future. We made the observation that blue noise masks provide much better error
// diffusion than Hilbert or Morton order based approaches. Hence we decided to use blue noise masks. A lot of the
// above makes use of Cranley-Patterson rotations which according to a paper by Heitz et al. (2019) is detrimental
// to the quality of the samples. However, they demonstrated this in the context of Sobol sequences and for other
// sequences like rank-1 lattices like the golden ratio based sequence this is not the case as they talked about
// in a revision of said paper in a 2021 SIGGRAPH talk [Bel21]. Also the Owen scrambled Sobol sequence is generally known
// to produce the highest quality low-discrepancy random numbers, however, they have also shown that the golden ratio
// based sequence is not meaningfully worse. Hence, we decided to use the above in detail described
// Cranley-Patterson rotation approach for generating our random numbers in combination with a blue noise mask
// provided by Cristoph Peters (http://momentsingraphics.de/BlueNoise.html) and a golden ratio based Rank-1 lattice
// sequence. This approach was also demonstrated to provide very good results in [Wol22]
// and was called a "blue noise animated golden ratio" sequence.
// To obtain a direction vector, we use the Squares random number generator which is a very fast
// minimal state high quality random number generated. This random number generator was used in Luminary
// already before so it was the obvious choice. It also provides very nice statistical properties (Though
// we have no idea about what that exactly means). Since a lot of the use cases in Luminary are 2D random
// points, we work always with pairs of randoms numbers using a generalization of the golden ratio based Rank-1 lattice that uses
// the inverse of the plastic number and its square.
//
// Now some small implementation details. For one, we perform all our operations using uint32_ts because
// we are working only with numbers in [0,1) due to the (mod 1) and integer addition and multiplication map
// perfectly to such a scenario. This also avoids some precision issues can run into when computing
// i * \alpha (mod 1) for large i. The conversion from the integer to the float is done by constructing
// a float whose significand is given by the integer. The Squares random number generator normally produces
// 64 bit numbers from 64 bit inputs but since we only need 32 bit outputs and only have 32 bit inputs, we
// modified the method to work with 32 bits. Similarly, was done for the 64 bit -> 32 bit version which we
// modified into a 32 bit -> 16 bit generator. The seed of this generator in general use cases is
// a key of alternating 0s and 1s in binary plus the thread ID.
//

#define BLUENOISE_TEX_DIM 128
#define BLUENOISE_TEX_DIM_MASK 0x7F

enum QuasiRandomTargetAllocation : uint32_t {
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_DIR_CHOICE        = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_DIR               = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BOUNCE_TRANSPARENCY      = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_TRANSPARENCY       = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LENS                     = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LENS_BLADE               = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_VOLUME_DIST              = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_RUSSIAN_ROULETTE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAMERA_JITTER            = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAMERA_TIME              = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_STEP_OFFSET        = 3,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_STEP_COUNT         = 3,
  QUASI_RANDOM_TARGET_ALLOCATION_CLOUD_DIR                = 128,
  QUASI_RANDOM_TARGET_ALLOCATION_SKY_STEP_OFFSET          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_SKY_INSCATTERING_STEP    = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_MICROFACET          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_DIFFUSE             = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_REFRACTION          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_RIS_DIFFUSE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BSDF_RIS_REFRACTION      = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_LIGHT_ID             = (32 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_RAY_DIR              = (32 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_RESAMPLING           = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_RIS_LIGHT_TREE           = (32 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_INITIAL          = 128,
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_RESAMPLE         = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_CAUSTIC_SUN_DIR          = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_BSDF           = 2,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_BSDF               = (2 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_RAY            = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_LIGHT_SUN_RIS_RESAMPLING = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_LIGHT_TREE        = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_LIGHT_LIST        = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_LIGHT_POINT       = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_DISTANCE          = (32 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_PHASE             = (32 * 32),
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_VERTEX_COUNT      = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_RESAMPLING        = 1,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_VERTEX_RESAMPLING = 32,
  QUASI_RANDOM_TARGET_ALLOCATION_BRIDGE_VERTEX_DISTANCE   = (32 * 64)
} typedef QuasiRandomTargetAllocation;

#define QUASI_RANDOM_TARGET_ALLOC(name) (QUASI_RANDOM_TARGET_##name + QUASI_RANDOM_TARGET_ALLOCATION_##name)

enum QuasiRandomTarget : uint32_t {
  QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE        = 0,
  QUASI_RANDOM_TARGET_BOUNCE_DIR               = QUASI_RANDOM_TARGET_ALLOC(BOUNCE_DIR_CHOICE),
  QUASI_RANDOM_TARGET_LENS                     = QUASI_RANDOM_TARGET_ALLOC(BOUNCE_DIR),
  QUASI_RANDOM_TARGET_LENS_BLADE               = QUASI_RANDOM_TARGET_ALLOC(LENS),
  QUASI_RANDOM_TARGET_VOLUME_DIST              = QUASI_RANDOM_TARGET_ALLOC(LENS_BLADE),
  QUASI_RANDOM_TARGET_RUSSIAN_ROULETTE         = QUASI_RANDOM_TARGET_ALLOC(VOLUME_DIST),
  QUASI_RANDOM_TARGET_CAMERA_JITTER            = QUASI_RANDOM_TARGET_ALLOC(RUSSIAN_ROULETTE),
  QUASI_RANDOM_TARGET_CAMERA_TIME              = QUASI_RANDOM_TARGET_ALLOC(CAMERA_JITTER),
  QUASI_RANDOM_TARGET_CLOUD_STEP_OFFSET        = QUASI_RANDOM_TARGET_ALLOC(CAMERA_TIME),
  QUASI_RANDOM_TARGET_CLOUD_STEP_COUNT         = QUASI_RANDOM_TARGET_ALLOC(CLOUD_STEP_OFFSET),
  QUASI_RANDOM_TARGET_CLOUD_DIR                = QUASI_RANDOM_TARGET_ALLOC(CLOUD_STEP_COUNT),
  QUASI_RANDOM_TARGET_SKY_STEP_OFFSET          = QUASI_RANDOM_TARGET_ALLOC(CLOUD_DIR),
  QUASI_RANDOM_TARGET_SKY_INSCATTERING_STEP    = QUASI_RANDOM_TARGET_ALLOC(SKY_STEP_OFFSET),
  QUASI_RANDOM_TARGET_BSDF_MICROFACET          = QUASI_RANDOM_TARGET_ALLOC(SKY_INSCATTERING_STEP),
  QUASI_RANDOM_TARGET_BSDF_DIFFUSE             = QUASI_RANDOM_TARGET_ALLOC(BSDF_MICROFACET),
  QUASI_RANDOM_TARGET_BSDF_REFRACTION          = QUASI_RANDOM_TARGET_ALLOC(BSDF_DIFFUSE),
  QUASI_RANDOM_TARGET_BSDF_RIS_DIFFUSE         = QUASI_RANDOM_TARGET_ALLOC(BSDF_REFRACTION),
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
  QUASI_RANDOM_TARGET_BRIDGE_LIGHT_TREE        = QUASI_RANDOM_TARGET_ALLOC(LIGHT_SUN_RIS_RESAMPLING),
  QUASI_RANDOM_TARGET_BRIDGE_LIGHT_LIST        = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_LIGHT_TREE),
  QUASI_RANDOM_TARGET_BRIDGE_LIGHT_POINT       = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_LIGHT_LIST),
  QUASI_RANDOM_TARGET_BRIDGE_DISTANCE          = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_LIGHT_POINT),
  QUASI_RANDOM_TARGET_BRIDGE_PHASE             = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_DISTANCE),
  QUASI_RANDOM_TARGET_BRIDGE_VERTEX_COUNT      = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_PHASE),
  QUASI_RANDOM_TARGET_BRIDGE_RESAMPLING        = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_VERTEX_COUNT),
  QUASI_RANDOM_TARGET_BRIDGE_VERTEX_RESAMPLING = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_RESAMPLING),
  QUASI_RANDOM_TARGET_BRIDGE_VERTEX_DISTANCE   = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_VERTEX_RESAMPLING),

  QUASI_RANDOM_TARGET_COUNT = QUASI_RANDOM_TARGET_ALLOC(BRIDGE_VERTEX_DISTANCE)
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

// [Wol22]
// A. Wolfe, N. Morrical, T. Akenine-Möller and R. Ramamoorthi, "Spatiotemporal Blue Noise Masks"
// Eurographics Symposium on Rendering, pp. 117-126, 2022.

// [Bel21]
// Laurent Belcour and Eric Heitz, "Lessons Learned and Improvements when Building Screen-Space Samplers with Blue-Noise Error Distribution"
// ACM SIGGRAPH 2021 Talks, pp. 1-2, 2021.

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

// Integer fractions of the actual numbers
#define R1_PHI1 2654435769u /* 0.61803398875f */
#define R2_PHI1 3242174889u /* 0.7548776662f  */
#define R2_PHI2 2447445413u /* 0.56984029f    */

__device__ uint32_t random_r1(const uint32_t offset) {
  return offset * R1_PHI1;
}

// [Rob18]
__device__ uint2 random_r2(const uint32_t offset) {
  const uint32_t v1 = offset * R2_PHI1;
  const uint32_t v2 = offset * R2_PHI2;

  return make_uint2(v1, v2);
}

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

  uint2 quasi = random_r2(sequence_id);

  // 0s are detrimental, hence we fix the lowest bit to 1, shouldn't be an issue.
  quasi.x *= random_uint32_t_base(0xfcbd6e15, dimension_index) | 1;
  quasi.y *= random_uint32_t_base(0x4bf53ed9, dimension_index) | 1;

  const uint2 pixel_offset = random_r2(dimension_index);
  const uint2 blue_noise   = random_blue_noise_mask_2D(pixel.x + (pixel_offset.x >> 25), pixel.y + (pixel_offset.y >> 25));

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
