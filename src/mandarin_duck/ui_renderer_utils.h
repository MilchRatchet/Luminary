#ifndef MANDARIN_DUCK_UI_RENDERER_UTILS_H
#define MANDARIN_DUCK_UI_RENDERER_UTILS_H

#include <immintrin.h>

#include "utils.h"

#define MANDARIN_DUCK_X86_INTRINSICS

inline void test_render_color(
  uint8_t* dst, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t ld, uint32_t color, uint8_t* mask) {
  const uint32_t cols = width >> 3;
  const uint32_t rows = height;

  __m256i a        = _mm256_set1_epi32(color);
  __m256i all_ones = _mm256_set1_epi32(0xFFFFFFFF);

  dst = dst + 4 * x + y * ld;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      __m256i mask_value = _mm256_loadu_si256((__m256i*) (mask + col * 32 + row * 64));

      __m256i base = _mm256_loadu_si256((__m256i*) (dst + col * 32));

      __m256i blend_mask = _mm256_cmpgt_epi32(mask_value, all_ones);

      base = _mm256_and_si256(base, blend_mask);
      base = _mm256_or_si256(base, _mm256_and_si256(a, _mm256_xor_si256(blend_mask, all_ones)));

      _mm256_storeu_si256((__m256i*) (dst + col * 32), base);
    }

    dst = dst + ld;
  }
}

#endif /* MANDARIN_DUCK_UI_RENDERER_UTILS_H */
