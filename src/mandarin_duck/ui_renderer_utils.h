#ifndef MANDARIN_DUCK_UI_RENDERER_UTILS_H
#define MANDARIN_DUCK_UI_RENDERER_UTILS_H

#include <immintrin.h>

#include "utils.h"

#define MANDARIN_DUCK_X86_INTRINSICS

struct Color256 {
  union {
    LuminaryARGB8 pixel[8];
#ifdef MANDARIN_DUCK_X86_INTRINSICS
    __m256 _imm;
    __m256i _immi;
#endif
  };
} typedef Color256;
_STATIC_ASSERT(sizeof(Color256) == 32);

struct Color128 {
  union {
    LuminaryARGB8 pixel[4];
#ifdef MANDARIN_DUCK_X86_INTRINSICS
    __m128 _imm;
    __m128i _immi;
#endif
  };
} typedef Color128;
_STATIC_ASSERT(sizeof(Color128) == 16);

struct Color32 {
  union {
    LuminaryARGB8 pixel;
    uint32_t data;
#ifdef MANDARIN_DUCK_X86_INTRINSICS

#endif
  };
} typedef Color32;
_STATIC_ASSERT(sizeof(Color32) == 4);

////////////////////////////////////////////////////////////////////
// Intrinsics
////////////////////////////////////////////////////////////////////

inline Color256 color256_set_1(const uint32_t a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_set1_epi32(a)};
#else
  // TODO
#endif
}

inline Color256 color256_set_1_64(const uint64_t a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_set1_epi64x(a)};
#else
  // TODO
#endif
}

inline Color256 color256_set(
  const uint32_t v0, const uint32_t v1, const uint32_t v2, const uint32_t v3, const uint32_t v4, const uint32_t v5, const uint32_t v6,
  const uint32_t v7) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_set_epi32(v0, v1, v2, v3, v4, v5, v6, v7)};
#else
  // TODO
#endif
}

inline Color256 color256_load(const uint8_t* ptr) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_loadu_si256((__m256i*) ptr)};
#else
  // TODO
#endif
}

inline void color256_store(const uint8_t* ptr, const Color256 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  _mm256_storeu_si256((__m256i*) ptr, a._immi);
#else
  // TODO
#endif
}

inline Color256 color256_add32(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_add_epi32(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_sub32(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_sub_epi32(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_mul32(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_mullo_epi32(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_mul16(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_mullo_epi16(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_add8(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_add_epi8(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_and(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_and_si256(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_or(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_or_si256(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_avg8(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_avg_epu8(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_maddubs16(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_maddubs_epi16(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color256 color256_packus16(const Color256 a, const Color256 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_packus_epi16(a._immi, b._immi)};
#else
  // TODO
#endif
}

#ifdef MANDARIN_DUCK_X86_INTRINSICS
#define color256_shift_right(__a, __count) ((Color256){._immi = _mm256_srli_epi32(__a._immi, __count)})
#define color256_shift_right16(__a, __count) ((Color256){._immi = _mm256_srli_epi16(__a._immi, __count)})
#define color256_shift_right64(__a, __count) ((Color256){._immi = _mm256_srli_epi64(__a._immi, __count)})
#define color256_shift_left(__a, __count) ((Color256){._immi = _mm256_slli_epi32(__a._immi, __count)})
#define color256_shift_left64(__a, __count) ((Color256){._immi = _mm256_slli_epi64(__a._immi, __count)})

#define color256_shuffle(__a, __imm) ((Color256){._immi = _mm256_shuffle_epi32(__a._immi, __imm)})
#define color256_shuffle8(__a, __b) ((Color256){._immi = _mm256_shuffle_epi8(__a._immi, __b._immi)})

#define color256_permute128(__a, __b, __imm) ((Color256){._immi = _mm256_permute2x128_si256(__a._immi, __b._immi, __imm)})
#endif

inline Color256 color256_load_si32(const uint8_t* ptr) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_castsi128_si256(_mm_cvtsi32_si128(*(uint32_t*) ptr))};
#else
  // TODO
#endif
}

inline Color256 color256_load_si128(const uint8_t* ptr) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) ptr))};
#else
  // TODO
#endif
}

inline void color256_store_si32(uint8_t* ptr, const Color256 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  *(int32_t*) ptr = _mm_cvtsi128_si32(_mm256_castsi256_si128(a._immi));
#else
  // TODO
#endif
}

inline void color256_store_si128(uint8_t* ptr, const Color256 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  _mm_storeu_si128((__m128i*) ptr, _mm256_castsi256_si128(a._immi));
#else
  // TODO
#endif
}

inline Color128 color128_load(const uint8_t* ptr) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_loadu_si128((__m128i*) ptr)};
#else
  // TODO
#endif
}

inline Color256 color128_extend(const Color128 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color256){._immi = _mm256_cvtepi32_epi64(a._immi)};
#else
  // TODO
#endif
}

inline Color128 color128_set_1(const uint32_t a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_set1_epi32(a)};
#else
  // TODO
#endif
}

inline Color128 color128_setzero() {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_setzero_si128()};
#else
  // TODO
#endif
}

inline Color128 color128_zero_extend8(Color128 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_cvtepu8_epi16(a._immi)};
#else
  // TODO
#endif
}

inline Color128 color128_adds16(Color128 a, Color128 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_adds_epu16(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color128 color128_subs16(Color128 a, Color128 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_subs_epu16(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline Color128 color128_packus16(Color128 a, Color128 b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return (Color128){._immi = _mm_packus_epi16(a._immi, b._immi)};
#else
  // TODO
#endif
}

inline void color128_store_stream32(uint8_t* ptr, uint32_t b) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  _mm_stream_si32((int*) ptr, (int32_t) b);
#else
  // TODO
#endif
}

inline uint32_t color128_get_low(Color128 a) {
#ifdef MANDARIN_DUCK_X86_INTRINSICS
  return _mm_cvtsi128_si32(a._immi);
#else
  // TODO
#endif
}

#ifdef MANDARIN_DUCK_X86_INTRINSICS
#define color128_shift_right16(__a, __count) ((Color128){._immi = _mm_srli_epi16(__a._immi, __count)})
#endif

////////////////////////////////////////////////////////////////////
// Macro functions
////////////////////////////////////////////////////////////////////

/*
 * @param alpha must be packed into the lower 8 bits of each 32 bit entry
 * @param mask_low16 Must contain 0x00FF00FF in each entry
 * @param mask_high16 Must contain 0xFF00FF00 in each entry
 * @param mask_add Must contain 0x00800080 in each entry
 */
inline Color256 color256_alpha_mul(
  const Color256 color, const Color256 alpha, const Color256 mask_low16, const Color256 mask_high16, const Color256 mask_add) {
  Color256 rb = color256_and(color, mask_low16);
  Color256 ga = color256_and(color256_shift_right(color, 8), mask_low16);

  rb = color256_mul32(rb, alpha);
  ga = color256_mul32(ga, alpha);

  rb = color256_add32(rb, mask_add);
  ga = color256_add32(ga, mask_add);

  Color256 rb_add = color256_shift_right(rb, 8);
  Color256 ga_add = color256_shift_right(ga, 8);

  rb_add = color256_and(rb_add, mask_low16);
  ga_add = color256_and(ga_add, mask_low16);

  rb = color256_add32(rb, rb_add);
  ga = color256_add32(ga, ga_add);

  rb = color256_and(rb, mask_high16);
  ga = color256_and(ga, mask_high16);

  return color256_or(ga, color256_shift_right(rb, 8));
}

/*
 * @param alpha must be packed into the lower 8 bits of each 32 bit entry
 * @param mask_low16 Must contain 0x00FF00FF in each entry
 * @param mask_high16 Must contain 0xFF00FF00 in each entry
 * @param mask_add Must contain 0x00800080 in each entry
 * @param mask_full_alpha Must contain 0x000000FF in each entry
 */
inline Color256 color256_alpha_blend(
  const Color256 color1, const Color256 color2, const Color256 alpha, const Color256 mask_low16, const Color256 mask_high16,
  const Color256 mask_add, const Color256 mask_full_alpha) {
  const Color256 inverse_alpha = color256_sub32(mask_full_alpha, alpha);

  const Color256 color1_alpha_mul = color256_alpha_mul(color1, alpha, mask_low16, mask_high16, mask_add);
  const Color256 color2_alpha_mul = color256_alpha_mul(color2, inverse_alpha, mask_low16, mask_high16, mask_add);

  return color256_add8(color1_alpha_mul, color2_alpha_mul);
}

inline uint64_t element_hash_accumulate(uint64_t hash, uint64_t value) {
  return _mm_crc32_u64(hash, value);
}

#endif /* MANDARIN_DUCK_UI_RENDERER_UTILS_H */
