#include "UI_blit.h"

#include <immintrin.h>
#include <string.h>

#include "UI.h"

#define BG_RED 20
#define BG_GREEN 30
#define BG_BLUE 40

#define BG_RED_H 90
#define BG_GREEN_H 90
#define BG_BLUE_H 90

void blit_gray(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t value) {
  height = min(height, hd - y);
  width  = min(width, ldd - x);
  for (int i = 0; i < height; i++) {
    memset(dst + (y + i) * 4 * ldd + 4 * x, value, 4 * width);
  }
}

void blit_color(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t red, uint8_t green, uint8_t blue) {
  height = min(height, hd - y);
  width  = min(width, ldd - x);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j]     = blue;
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 1] = green;
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 2] = red;
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 3] = 0;
    }
  }
}

void blit_color_shaded(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t red, uint8_t green, uint8_t blue) {
  height = min(height, hd - y);
  width  = min(width, ldd - x);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const uint8_t value_b                      = dst[(y + i) * 4 * ldd + 4 * x + 4 * j];
      const uint8_t value_g                      = dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 1];
      const uint8_t value_r                      = dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 2];
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j]     = min(blue, value_b);
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 1] = min(green, value_g);
      dst[(y + i) * 4 * ldd + 4 * x + 4 * j + 2] = min(red, value_r);
    }
  }
}

#if defined(__AVX2__)
void blit_UI_internal(UI* ui, uint8_t* target, int width, int height) {
  const int k      = UI_WIDTH / 8;
  const int offset = ui->scroll_pos % PANEL_HEIGHT;

  const __m256i bg = _mm256_setr_epi8(
    BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE,
    BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0);
  const __m256i bg_h = _mm256_setr_epi8(
    BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H,
    BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H,
    BG_GREEN_H, BG_RED_H, 0);

  if (ui->border_hover) {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 4 * width + 4 * ui->x + j * 32;

        __m256i pixel = _mm256_loadu_si256((__m256i*) (target + target_index));
        pixel         = _mm256_avg_epu8(bg_h, pixel);
        _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
      }
    }
  }
  else {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 4 * width + 4 * ui->x + j * 32;

        __m256i pixel = _mm256_loadu_si256((__m256i*) (target + target_index));
        pixel         = _mm256_avg_epu8(bg, pixel);
        _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
      }
    }
  }

  if (ui->panel_hover == 0) {
    for (int i = 0; i < PANEL_HEIGHT; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 32;
        const int ui_index     = j * 32 + i * UI_WIDTH * 4;

        __m256i ui_pixel = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
        __m256i mask     = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
        __m256i pixel    = _mm256_loadu_si256((__m256i*) (target + target_index));
        pixel            = _mm256_avg_epu8(bg_h, pixel);
        pixel            = _mm256_blendv_epi8(pixel, ui_pixel, mask);
        _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
      }
    }
  }
  else {
    for (int i = 0; i < PANEL_HEIGHT; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 32;
        const int ui_index     = j * 32 + i * UI_WIDTH * 4;

        __m256i ui_pixel = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
        __m256i mask     = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
        __m256i pixel    = _mm256_loadu_si256((__m256i*) (target + target_index));
        pixel            = _mm256_avg_epu8(bg, pixel);
        pixel            = _mm256_blendv_epi8(pixel, ui_pixel, mask);
        _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
      }
    }
  }

  const int hover_start = ui->panel_hover * PANEL_HEIGHT - ui->scroll_pos;
  const int hover_end   = min((ui->panel_hover + 1) * PANEL_HEIGHT - ui->scroll_pos, UI_HEIGHT);

  int i = PANEL_HEIGHT;

  for (; i < hover_start; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 32;
      const int ui_index     = j * 32 + (i + offset) * UI_WIDTH * 4;

      __m256i ui_pixel = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask     = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i pixel    = _mm256_loadu_si256((__m256i*) (target + target_index));
      pixel            = _mm256_avg_epu8(bg, pixel);
      pixel            = _mm256_blendv_epi8(pixel, ui_pixel, mask);
      _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
    }
  }

  for (; i < hover_end; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 32;
      const int ui_index     = j * 32 + (i + offset) * UI_WIDTH * 4;

      __m256i ui_pixel = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask     = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i pixel    = _mm256_loadu_si256((__m256i*) (target + target_index));
      pixel            = _mm256_avg_epu8(bg_h, pixel);
      pixel            = _mm256_blendv_epi8(pixel, ui_pixel, mask);
      _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
    }
  }

  for (; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 32;
      const int ui_index     = j * 32 + (i + offset) * UI_WIDTH * 4;

      __m256i ui_pixel = _mm256_loadu_si256((__m256i*) (ui->pixels + ui_index));
      __m256i mask     = _mm256_loadu_si256((__m256i*) (ui->pixels_mask + ui_index));
      __m256i pixel    = _mm256_loadu_si256((__m256i*) (target + target_index));
      pixel            = _mm256_avg_epu8(bg, pixel);
      pixel            = _mm256_blendv_epi8(pixel, ui_pixel, mask);
      _mm256_storeu_si256((__m256i*) (target + target_index), pixel);
    }
  }
}
#else
void blit_UI_internal(UI* ui, uint8_t* target, int width, int height, int offset) {
  const int k = UI_WIDTH / 4;
  const __m128i bg =
    _mm_setr_epi8(BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0, BG_BLUE, BG_GREEN, BG_RED, 0);
  const __m128i bg_h = _mm_setr_epi8(
    BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H, BG_RED_H, 0, BG_BLUE_H, BG_GREEN_H,
    BG_RED_H, 0);

  if (ui->border_hover) {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 4 * width + 4 * ui->x + j * 16;

        __m128i pixel = _mm_loadu_si128((__m128i*) (target + target_index));
        pixel         = _mm_avg_epu8(bg_h, pixel);
        _mm_storeu_si128((__m128i*) (target + target_index), pixel);
      }
    }
  }
  else {
    for (int i = 0; i < UI_BORDER_SIZE; i++) {
      for (int j = 0; j < k; j++) {
        const int target_index = (i + ui->y) * 4 * width + 4 * ui->x + j * 16;

        __m128i pixel = _mm_loadu_si128((__m128i*) (target + target_index));
        pixel         = _mm_avg_epu8(bg, pixel);
        _mm_storeu_si128((__m128i*) (target + target_index), pixel);
      }
    }
  }

  const int hover_start = ui->panel_hover * PANEL_HEIGHT;
  const int hover_end   = (ui->panel_hover + 1) * PANEL_HEIGHT;

  int i = 0;

  for (; i < hover_start; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 16;
      const int ui_index     = j * 16 + (i + offset) * UI_WIDTH * 4;

      __m128i ui_pixel = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask     = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i pixel    = _mm_loadu_si128((__m128i*) (target + target_index));
      pixel            = _mm_avg_epu8(bg, pixel);
      pixel            = _mm_blendv_epi8(pixel, ui_pixel, mask);
      _mm_storeu_si128((__m128i*) (target + target_index), pixel);
    }
  }

  for (; i < hover_end; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 16;
      const int ui_index     = j * 16 + (i + offset) * UI_WIDTH * 4;

      __m128i ui_pixel = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask     = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i pixel    = _mm_loadu_si128((__m128i*) (target + target_index));
      pixel            = _mm_avg_epu8(bg_h, pixel);
      pixel            = _mm_blendv_epi8(pixel, ui_pixel, mask);
      _mm_storeu_si128((__m128i*) (target + target_index), pixel);
    }
  }

  for (; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + UI_BORDER_SIZE + ui->y) * 4 * width + 4 * ui->x + j * 16;
      const int ui_index     = j * 16 + (i + offset) * UI_WIDTH * 4;

      __m128i ui_pixel = _mm_loadu_si128((__m128i*) (ui->pixels + ui_index));
      __m128i mask     = _mm_loadu_si128((__m128i*) (ui->pixels_mask + ui_index));
      __m128i pixel    = _mm_loadu_si128((__m128i*) (target + target_index));
      pixel            = _mm_avg_epu8(bg, pixel);
      pixel            = _mm_blendv_epi8(pixel, ui_pixel, mask);
      _mm_storeu_si128((__m128i*) (target + target_index), pixel);
    }
  }
}
#endif

void blit_text(UI* ui, SDL_Surface* text, int x, int y, int ldd, int hd) {
  uint8_t* text_pixels = (uint8_t*) text->pixels;

  int height = min(text->h, hd - y);
  int width  = min(text->w, ldd - x);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < 4 * width; j++) {
      const int ui_index        = 4 * x + j + (i + y) * ldd * 4;
      const float alpha         = text_pixels[j / 4 + i * text->pitch] / 255.0f;
      const uint8_t ui_pixel    = ui->pixels[ui_index];
      ui->pixels[ui_index]      = alpha * 255.0f + (1.0f - alpha) * ui_pixel;
      ui->pixels_mask[ui_index] = (alpha > 0.0f) ? 0xff : 0;
    }
  }
}
