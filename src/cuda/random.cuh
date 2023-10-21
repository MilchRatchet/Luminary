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
// very close to each other. If we now think about integrating a function and another classical approach
// of evaluating the function at equidistant points then we see that the equidistant points manage to cover
// a large range of the functions domain and will only fail if the function has very different outputs
// inbetween those points. Our clumped up random points however will fail to give a good representation of our
// function even if the function is well behaved since we are not effectively covering the functions domain.
// Hence we use low-discrepancy random numbers which are uniformly distributed but neighbouring numbers
// in the sequence are far apart from another. The analogue to the equidistant points is then a sequence
// of points that each split the existing sampling gaps in half, i.e., 1/2, 1/4, 3/4, 1/8, 5/8 and so on.
// This sequence is called the "van der Corput" sequence. While the points are low-discrepency, they
// have a visible pattern and hence more "randomly" distributed points are to be preferred.
// A multitude of random number sequences have been proposed one of the most notable is the Halton and
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
// a D-dimensional random number by computing one random number and shifting it using a different sequence
// of random numbers. All entries inside this D-dimensional random number must be uncorrelated or else we would
// obtain a biased result. You can think of this as we want the decision to use a diffuse importance sampled
// direction to not be correlated with the actual sampled direction obtained from that method, i.e., we don't
// want all diffuse samples to go up. Hence we shift them using a uncorrelated sequence of random numbers.
// We perform this shift as r + s_j (mod 1) where s_j is our shift number and r is our low-discrepancy random number.
// This is also called a Cranley-Patterson rotation. In order for the generated D-dimensional random numbers to stay
// low-discrepancy, we have to apply the same shift s to each low-discrepancy random number.
//
// Now that we can generate low-discrepancy D-dimensional random numbers to perform quasi-Monte Carlo integration for
// each pixel, we need to think about the sequences for all our pixels. If we were to use the same random number
// sequence for each pixel, we would obtain almost identical errors in each pixel. On the surface this does not sound
// too bad but visually this means that the error is much more prominent and the image after a few samples per pixel
// will not look anything like the final converged image. To fix this we can apply the same approach as above by
// applying another shift to the D-dimensional low-discrepancy random numbers that is different for each pixel.
// This will distribute the error across pixels which gives a much more visually pleasing result. In fact, this
// is the point where our images start to become noisy, without this they would look almost noise free but heavily biased
// instead. It is also much clearer to understand if an image is (pretty much) converged based on the noise levels
// as it is impossible to visually identify the error if no reference image exists. For the choice of the shift number
// we could use standard uncorrelated random numbers but if we do that we would run into clumping again which means
// that the error would not be evenly distributed over the image. Hence we again use low-discrepancy numbers to perform
// this shift. A common approach for this are so called blue noise masks which are specifically crafted images that shift
// each sequence of D-dimensional random numbers such that the error is very evenly distributed across pixels. Alternatively,
// other approach likes using Hilbert or Morton indexing together with scrambling have been shown to be a good alternative that
// don't need any precomputed texture. They work by having a very large base sequence of low-discrepancy numbers and
// computing a different index in this sequence for each pixel.
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
// For the uncorrelated shifts we use the Squares random number generator which is a very fast
// minimal state high quality random number generated. This random number generator was used in Luminary
// already before so it was the obvious choice. It also provides very nice statistical properties (Though
// we have no idea about what that exactly means). Since a lot of the use cases in Luminary are 2D random
// points, we also included a generalization of the golden ratio based Rank-1 lattice that uses
// the inverse of the plastic number and its square. This is also the only point where we are unsure whether
// we introduce some correlation issues because we use both the 1D and the 2D sequence at the same time, albeit
// for different decision. It is unclear to us whether these two sequence may cause any issues when used like that,
// however, we couldn't observe any issues and comparisons to renders with the previous white noise random
// number generator demonstrated that we converge to the same result. Ultimately, each random number is given by
//
//                    r(t) + b(x,y) + w_1(d) + w_2(u) (mod 1)
//
// where r is our low-discrepancy sequence, b is the blue noise mask and w_1 and w_2 are differently seeded
// white noise generators. Further, t is the sampled ID (also called the temporal frame in Luminary),
// x and y are the pixel coordinates, d is the current depth of the path and u is the use case of this random number.
// Each usage of these numbers is mapped out using an enum, this makes sure that no two decisions use the same
// random number and it is nice to work with. Important to note is that this random number generator is not
// counter based, we do not have to keep track of the number of random numbers generated.
//
// Now some small implementation details. For one, we perform all our operations using uint32_ts because
// we are working only with numbers in [0,1) due to the (mod 1) and integer addition and multiplication map
// perfectly to such a scenario. This also avoids some precision issues can run into when computing
// i * \alpha (mod 1) for large i. The conversion from the integer to the float is done by constructing
// a float whose significand is given by the integer. The Squares random number generator normally produces
// 64 bit numbers from 64 bit inputs but since we only need 32 bit outputs and only have 32 bit inputs, we
// modified the method to work with 32 bits. Similarly, was done for the 64 bit -> 32 bit version which we
// modified into a 32 bit -> 16 bit generator. The seed of this generator in general use cases is
// a key of alternating 0s and 1s in binary plus the thread ID. For our low-discrepancy random number
// we used a seed that instead of adding the thread ID, adds a key as provided by the original Squares
// RNG paper [Wid20].
//

enum QuasiRandomTarget : uint32_t {
  QUASI_RANDOM_TARGET_BOUNCE_DIR_CHOICE           = 0,   /* 1 */
  QUASI_RANDOM_TARGET_BOUNCE_DIR                  = 1,   /* 2 */
  QUASI_RANDOM_TARGET_BOUNCE_TRANSPARENCY         = 3,   /* 1 */
  QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY          = 4,   /* 1 */
  QUASI_RANDOM_TARGET_LIGHT_TRANSPARENCY_ROULETTE = 5,   /* 1 */
  QUASI_RANDOM_TARGET_RESTIR_CHOICE               = 6,   /* 2 + 128 */
  QUASI_RANDOM_TARGET_RESTIR_DIR                  = 136, /* 2 + 128 */
  QUASI_RANDOM_TARGET_RESTIR_GENERATION           = 266, /* 128 */
  QUASI_RANDOM_TARGET_LENS                        = 394, /* 2 */
  QUASI_RANDOM_TARGET_VOLUME_DIST                 = 396, /* 1 */
  QUASI_RANDOM_TARGET_RUSSIAN_ROULETTE            = 397, /* 1 */
  QUASI_RANDOM_TARGET_CLOUD_STEP_OFFSET           = 398, /* 3 */
  QUASI_RANDOM_TARGET_CLOUD_STEP_COUNT            = 401  /* 3 */
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
// Random number generators
////////////////////////////////////////////////////////////////////

// Integer fractions of the actual numbers
#define R1_PHI 2654435768u  /* 0.61803398875f */
#define R2_PHI1 3242174888u /* 0.7548776662f  */
#define R2_PHI2 2447445413u /* 0.56984029f    */

__device__ uint32_t random_r1(const uint32_t offset) {
  const uint32_t v = offset * R1_PHI;

  return v;
}

__device__ uint2 random_r2(const uint32_t offset) {
  const uint32_t v1 = offset * R2_PHI1;
  const uint32_t v2 = offset * R2_PHI2;

  return make_uint2(v1, v2);
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

__device__ uint32_t random_blue_noise_mask_1D(const uint32_t pixel) {
  const uint32_t y = pixel / device.width;
  const uint32_t x = pixel - y * device.width;

  const DeviceTexture tex   = *device.ptrs.bluenoise_1D_tex;
  const uint32_t blue_noise = tex2D<uint32_t>(tex.tex, x * tex.inv_width, y * tex.inv_height);

  return blue_noise << 16;
}

__device__ uint2 random_blue_noise_mask_2D(const uint32_t pixel) {
  const uint32_t y = pixel / device.width;
  const uint32_t x = pixel - y * device.width;

  const DeviceTexture tex = *device.ptrs.bluenoise_2D_tex;
  const uint2 blue_noise  = tex2D<uint2>(tex.tex, x * tex.inv_width, y * tex.inv_height);

  return make_uint2(blue_noise.x << 16, blue_noise.y << 16);
}

__device__ uint32_t random_target_offset_1D(const uint32_t target) {
  const uint32_t depth_offset  = random_uint32_t_base(0xc8e4fd15, device.depth);
  const uint32_t target_offset = random_uint32_t_base(0x4ce32f6d, target);

  return depth_offset + target_offset;
}

__device__ uint2 random_target_offset_2D(const uint32_t target) {
  const uint32_t depth_offset1  = random_uint32_t_base(0xfcbd6e15, 2 * device.depth);
  const uint32_t target_offset1 = random_uint32_t_base(0x4bf53ed9, target);

  const uint32_t depth_offset2  = random_uint32_t_base(0xfcbd6e15, 2 * device.depth + 1);
  const uint32_t target_offset2 = random_uint32_t_base(0x4bf53ed9, target + 1);

  return make_uint2(depth_offset1 + target_offset1, depth_offset2 + target_offset2);
}

__device__ float quasirandom_sequence_1D(const uint32_t target, const uint32_t pixel) {
  uint32_t quasi = random_r1(device.temporal_frames);

  quasi += random_blue_noise_mask_1D(pixel);
  quasi += random_target_offset_1D(target);

  return random_uint32_t_to_float(quasi);
}

// This is a special version of the quasirandom sequence that is not low discrepency temporally but instead
// allows for a custom sequence id. This is useful for things like ReSTIR where the light selection
// does not need to vary between frames as the list we are indexing changes every frame. Instead, we
// want to make sure that we sample unique lights.
__device__ float quasirandom_sequence_1D_intraframe(const uint32_t target, const uint32_t pixel, const uint32_t sequence_id) {
  uint32_t quasi = random_r1(sequence_id);

  quasi += random_blue_noise_mask_1D(pixel);
  quasi += random_target_offset_1D(target);

  return random_uint32_t_to_float(quasi);
}

__device__ float2 quasirandom_sequence_2D(const uint32_t target, const uint32_t pixel) {
  uint2 quasi = random_r2(device.temporal_frames);

  const uint2 blue_noise_mask     = random_blue_noise_mask_2D(pixel);
  const uint2 depth_target_offset = random_target_offset_2D(target);

  quasi.x += depth_target_offset.x;
  quasi.y += depth_target_offset.y;

  quasi.x += blue_noise_mask.x;
  quasi.y += blue_noise_mask.y;

  return make_float2(random_uint32_t_to_float(quasi.x), random_uint32_t_to_float(quasi.y));
}

__device__ float random_dither_mask(const uint32_t x, const uint32_t y) {
  DeviceTexture tex         = *device.ptrs.bluenoise_1D_tex;
  const uint16_t blue_noise = tex2D<uint16_t>(tex.tex, x * tex.inv_width, y * tex.inv_height);

  return random_uint16_t_to_float(blue_noise);
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
