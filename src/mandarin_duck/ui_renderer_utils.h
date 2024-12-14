#ifndef MANDARIN_DUCK_UI_RENDERER_UTILS_H
#define MANDARIN_DUCK_UI_RENDERER_UTILS_H

#include <immintrin.h>

#include "utils.h"

#define MANDARIN_DUCK_X86_INTRINSICS

inline void test_render_color(uint8_t* dst, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t ld, uint32_t color) {
  const uint32_t cols = width >> 3;
  const uint32_t rows = height;

  __m256i a = _mm256_set1_epi32(color);

  dst = dst + 4 * x + y * ld;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      _mm256_storeu_si256((__m256i*) (dst + col * 32), a);
    }

    dst = dst + ld;
  }
}

#endif /* MANDARIN_DUCK_UI_RENDERER_UTILS_H */
