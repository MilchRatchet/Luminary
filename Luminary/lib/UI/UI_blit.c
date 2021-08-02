#include <string.h>
#include <immintrin.h>
#include "UI.h"
#include "UI_blit.h"

#define BG_RED 20
#define BG_GREEN 30
#define BG_BLUE 40

#define BG_RED_H 90
#define BG_GREEN_H 90
#define BG_BLUE_H 90

void blit_fast(uint8_t* src, int lds, uint8_t* dst, int ldd, int width, int height) {
  for (int i = 0; i < height; i++) {
    memcpy(dst + i * 3 * ldd, src + i * 3 * lds, 3 * width);
  }
}

void blit_gray(uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t value) {
  for (int i = 0; i < height; i++) {
    memset(dst + (y + i) * 3 * ldd + 3 * x, value, 3 * width);
  }
}

void blit_color(
  uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t red, uint8_t green,
  uint8_t blue) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j]     = red;
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 1] = green;
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 2] = blue;
    }
  }
}

void blit_color_shaded(
  uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t red, uint8_t green,
  uint8_t blue) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const uint8_t value_r                      = dst[(y + i) * 3 * ldd + 3 * x + 3 * j];
      const uint8_t value_g                      = dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 1];
      const uint8_t value_b                      = dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 2];
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j]     = min(red, value_r);
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 1] = min(green, value_g);
      dst[(y + i) * 3 * ldd + 3 * x + 3 * j + 2] = min(blue, value_b);
    }
  }
}

#if defined(__AVX2__)
void blit_UI_internal(UI* ui, uint8_t* target, int width, int height) {
  const int k       = UI_WIDTH / 32;
  const __m256i bg1 = _mm256_setr_epi8(
    BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED,
    BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN,
    BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE,
    BG_RED, BG_GREEN);
  const __m256i bg2 = _mm256_setr_epi8(
    BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE,
    BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED,
    BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN,
    BG_BLUE, BG_RED);
  const __m256i bg3 = _mm256_setr_epi8(
    BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN,
    BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE,
    BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED,
    BG_GREEN, BG_BLUE);

  const __m256i bg1_h = _mm256_setr_epi8(
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H,
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H,
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H,
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H);
  const __m256i bg2_h = _mm256_setr_epi8(
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H,
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H,
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H,
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H);
  const __m256i bg3_h = _mm256_setr_epi8(
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H,
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H,
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H,
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H);

  if (ui->border_hover) {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 96;

        __m256i target1 = _mm256_loadu_si256((__m256i*) (target + target_index));
        target1         = _mm256_avg_epu8(bg1_h, target1);
        _mm256_storeu_si256((__m256i*) (target + target_index), target1);
        __m256i target2 = _mm256_loadu_si256((__m256i*) (target + target_index + 32));
        target2         = _mm256_avg_epu8(bg2_h, target2);
        _mm256_storeu_si256((__m256i*) (target + target_index + 32), target2);
        __m256i target3 = _mm256_loadu_si256((__m256i*) (target + target_index + 64));
        target3         = _mm256_avg_epu8(bg3_h, target3);
        _mm256_storeu_si256((__m256i*) (target + target_index + 64), target3);
      }
    }
  }
  else {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 96;

        __m256i target1 = _mm256_loadu_si256((__m256i*) (target + target_index));
        target1         = _mm256_avg_epu8(bg1, target1);
        _mm256_storeu_si256((__m256i*) (target + target_index), target1);
        __m256i target2 = _mm256_loadu_si256((__m256i*) (target + target_index + 32));
        target2         = _mm256_avg_epu8(bg2, target2);
        _mm256_storeu_si256((__m256i*) (target + target_index + 32), target2);
        __m256i target3 = _mm256_loadu_si256((__m256i*) (target + target_index + 64));
        target3         = _mm256_avg_epu8(bg3, target3);
        _mm256_storeu_si256((__m256i*) (target + target_index + 64), target3);
      }
    }
  }

  const int hover_start = ui->panel_hover * PANEL_HEIGHT;
  const int hover_end   = (ui->panel_hover + 1) * PANEL_HEIGHT;

  int i = 0;

  for (; i < hover_start; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 96;
      const int ui_index     = j * 96 + i * UI_WIDTH * 3;

      __m256i ui1     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask1   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i target1 = _mm256_loadu_si256((__m256i*) (target + target_index));
      target1         = _mm256_avg_epu8(bg1, target1);
      target1         = _mm256_blendv_epi8(target1, ui1, mask1);
      _mm256_storeu_si256((__m256i*) (target + target_index), target1);
      __m256i ui2     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 32));
      __m256i mask2   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 32));
      __m256i target2 = _mm256_loadu_si256((__m256i*) (target + target_index + 32));
      target2         = _mm256_avg_epu8(bg2, target2);
      target2         = _mm256_blendv_epi8(target2, ui2, mask2);
      _mm256_storeu_si256((__m256i*) (target + target_index + 32), target2);
      __m256i ui3     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 64));
      __m256i mask3   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 64));
      __m256i target3 = _mm256_loadu_si256((__m256i*) (target + target_index + 64));
      target3         = _mm256_avg_epu8(bg3, target3);
      target3         = _mm256_blendv_epi8(target3, ui3, mask3);
      _mm256_storeu_si256((__m256i*) (target + target_index + 64), target3);
    }
  }

  for (; i < hover_end; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 96;
      const int ui_index     = j * 96 + i * UI_WIDTH * 3;

      __m256i ui1     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask1   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i target1 = _mm256_loadu_si256((__m256i*) (target + target_index));
      target1         = _mm256_avg_epu8(bg1_h, target1);
      target1         = _mm256_blendv_epi8(target1, ui1, mask1);
      _mm256_storeu_si256((__m256i*) (target + target_index), target1);
      __m256i ui2     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 32));
      __m256i mask2   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 32));
      __m256i target2 = _mm256_loadu_si256((__m256i*) (target + target_index + 32));
      target2         = _mm256_avg_epu8(bg2_h, target2);
      target2         = _mm256_blendv_epi8(target2, ui2, mask2);
      _mm256_storeu_si256((__m256i*) (target + target_index + 32), target2);
      __m256i ui3     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 64));
      __m256i mask3   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 64));
      __m256i target3 = _mm256_loadu_si256((__m256i*) (target + target_index + 64));
      target3         = _mm256_avg_epu8(bg3_h, target3);
      target3         = _mm256_blendv_epi8(target3, ui3, mask3);
      _mm256_storeu_si256((__m256i*) (target + target_index + 64), target3);
    }
  }

  for (; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 96;
      const int ui_index     = j * 96 + i * UI_WIDTH * 3;

      __m256i ui1     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask1   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i target1 = _mm256_loadu_si256((__m256i*) (target + target_index));
      target1         = _mm256_avg_epu8(bg1, target1);
      target1         = _mm256_blendv_epi8(target1, ui1, mask1);
      _mm256_storeu_si256((__m256i*) (target + target_index), target1);
      __m256i ui2     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 32));
      __m256i mask2   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 32));
      __m256i target2 = _mm256_loadu_si256((__m256i*) (target + target_index + 32));
      target2         = _mm256_avg_epu8(bg2, target2);
      target2         = _mm256_blendv_epi8(target2, ui2, mask2);
      _mm256_storeu_si256((__m256i*) (target + target_index + 32), target2);
      __m256i ui3     = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index + 64));
      __m256i mask3   = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index + 64));
      __m256i target3 = _mm256_loadu_si256((__m256i*) (target + target_index + 64));
      target3         = _mm256_avg_epu8(bg3, target3);
      target3         = _mm256_blendv_epi8(target3, ui3, mask3);
      _mm256_storeu_si256((__m256i*) (target + target_index + 64), target3);
    }
  }
}
#else
void blit_UI_internal(UI* ui, uint8_t* target, int width, int height) {
  const int k       = UI_WIDTH / 16;
  const __m128i bg1 = _mm_setr_epi8(
    BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED,
    BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED);
  const __m128i bg2 = _mm_setr_epi8(
    BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN,
    BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN);
  const __m128i bg3 = _mm_setr_epi8(
    BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE,
    BG_RED, BG_GREEN, BG_BLUE, BG_RED, BG_GREEN, BG_BLUE);
  const __m128i bg1_h = _mm_setr_epi8(
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H,
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H);
  const __m128i bg2_h = _mm_setr_epi8(
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H,
    BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H);
  const __m128i bg3_h = _mm_setr_epi8(
    BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H,
    BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H, BG_RED_H, BG_GREEN_H, BG_BLUE_H);

  if (ui->border_hover) {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 48;

        __m128i target1 = _mm_loadu_si128((__m128i*) (target + target_index));
        target1         = _mm_avg_epu8(bg1_h, target1);
        _mm_storeu_si128((__m128i*) (target + target_index), target1);
        __m128i target2 = _mm_loadu_si128((__m128i*) (target + target_index + 16));
        target2         = _mm_avg_epu8(bg2_h, target2);
        _mm_storeu_si128((__m128i*) (target + target_index + 16), target2);
        __m128i target3 = _mm_loadu_si128((__m128i*) (target + target_index + 32));
        target3         = _mm_avg_epu8(bg3_h, target3);
        _mm_storeu_si128((__m128i*) (target + target_index + 32), target3);
      }
    }
  }
  else {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 48;

        __m128i target1 = _mm_loadu_si128((__m128i*) (target + target_index));
        target1         = _mm_avg_epu8(bg1, target1);
        _mm_storeu_si128((__m128i*) (target + target_index), target1);
        __m128i target2 = _mm_loadu_si128((__m128i*) (target + target_index + 16));
        target2         = _mm_avg_epu8(bg2, target2);
        _mm_storeu_si128((__m128i*) (target + target_index + 16), target2);
        __m128i target3 = _mm_loadu_si128((__m128i*) (target + target_index + 32));
        target3         = _mm_avg_epu8(bg3, target3);
        _mm_storeu_si128((__m128i*) (target + target_index + 32), target3);
      }
    }
  }

  const int hover_start = ui->panel_hover * PANEL_HEIGHT;
  const int hover_end   = (ui->panel_hover + 1) * PANEL_HEIGHT;

  int i = 0;

  for (; i < hover_start; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 48;
      const int ui_index     = j * 48 + i * UI_WIDTH * 3;

      __m128i ui1     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask1   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i target1 = _mm_loadu_si128((__m128i*) (target + target_index));
      target1         = _mm_avg_epu8(bg1, target1);
      target1         = _mm_blendv_epi8(target1, ui1, mask1);
      _mm_storeu_si128((__m128i*) (target + target_index), target1);
      __m128i ui2     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 16));
      __m128i mask2   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 16));
      __m128i target2 = _mm_loadu_si128((__m128i*) (target + target_index + 16));
      target2         = _mm_avg_epu8(bg2, target2);
      target2         = _mm_blendv_epi8(target2, ui2, mask2);
      _mm_storeu_si128((__m128i*) (target + target_index + 16), target2);
      __m128i ui3     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 32));
      __m128i mask3   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 32));
      __m128i target3 = _mm_loadu_si128((__m128i*) (target + target_index + 32));
      target3         = _mm_avg_epu8(bg3, target3);
      target3         = _mm_blendv_epi8(target3, ui3, mask3);
      _mm_storeu_si128((__m128i*) (target + target_index + 32), target3);
    }
  }

  for (; i < hover_end; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 48;
      const int ui_index     = j * 48 + i * UI_WIDTH * 3;

      __m128i ui1     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask1   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i target1 = _mm_loadu_si128((__m128i*) (target + target_index));
      target1         = _mm_avg_epu8(bg1_h, target1);
      target1         = _mm_blendv_epi8(target1, ui1, mask1);
      _mm_storeu_si128((__m128i*) (target + target_index), target1);
      __m128i ui2     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 16));
      __m128i mask2   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 16));
      __m128i target2 = _mm_loadu_si128((__m128i*) (target + target_index + 16));
      target2         = _mm_avg_epu8(bg2_h, target2);
      target2         = _mm_blendv_epi8(target2, ui2, mask2);
      _mm_storeu_si128((__m128i*) (target + target_index + 16), target2);
      __m128i ui3     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 32));
      __m128i mask3   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 32));
      __m128i target3 = _mm_loadu_si128((__m128i*) (target + target_index + 32));
      target3         = _mm_avg_epu8(bg3_h, target3);
      target3         = _mm_blendv_epi8(target3, ui3, mask3);
      _mm_storeu_si128((__m128i*) (target + target_index + 32), target3);
    }
  }

  for (; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 3 * width + 3 * ui->x + j * 48;
      const int ui_index     = j * 48 + i * UI_WIDTH * 3;

      __m128i ui1     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask1   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i target1 = _mm_loadu_si128((__m128i*) (target + target_index));
      target1         = _mm_avg_epu8(bg1, target1);
      target1         = _mm_blendv_epi8(target1, ui1, mask1);
      _mm_storeu_si128((__m128i*) (target + target_index), target1);
      __m128i ui2     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 16));
      __m128i mask2   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 16));
      __m128i target2 = _mm_loadu_si128((__m128i*) (target + target_index + 16));
      target2         = _mm_avg_epu8(bg2, target2);
      target2         = _mm_blendv_epi8(target2, ui2, mask2);
      _mm_storeu_si128((__m128i*) (target + target_index + 16), target2);
      __m128i ui3     = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index + 32));
      __m128i mask3   = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index + 32));
      __m128i target3 = _mm_loadu_si128((__m128i*) (target + target_index + 32));
      target3         = _mm_avg_epu8(bg3, target3);
      target3         = _mm_blendv_epi8(target3, ui3, mask3);
      _mm_storeu_si128((__m128i*) (target + target_index + 32), target3);
    }
  }
}
#endif
