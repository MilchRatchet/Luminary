#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "UI.h"
#include "UI_blur.h"
#include "UI_text.h"
#include "UI_panel.h"

#define BG_RED 20
#define BG_GREEN 30
#define BG_BLUE 40

static size_t compute_scratch_space() {
  size_t val = blur_scratch_needed();

  return val;
}

/*
 * Requires the font of the UI to be initialized.
 */
static UIPanel* create_general_panels(UI* ui) {
  UIPanel* panels = (UIPanel*) malloc(sizeof(UIPanel) * UI_PANELS_GENERAL_COUNT);

  panels[0] = create_slider(ui, "Far Clip Distance", 0);

  return panels;
}

UI init_UI() {
  UI ui;
  ui.x = 100;
  ui.y = 100;

  ui.pixels      = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  ui.pixels_mask = (uint8_t*) malloc(sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  size_t scratch_size = compute_scratch_space();
  ui.scratch          = malloc(scratch_size);
  init_text(&ui);

  ui.general_panels = create_general_panels(&ui);

  return ui;
}

void render_UI(UI* ui) {
  memset(ui->pixels, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);
  memset(ui->pixels_mask, 0, sizeof(uint8_t) * UI_WIDTH * UI_HEIGHT * 3);

  SDL_Surface* text = render_text(ui, "General");

  blit_text(ui, text, 50, 50);

  SDL_FreeSurface(text);
}

#if defined(__AVX2__)
static void blit_to_target(UI* ui, uint8_t* target, int width, int height) {
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
  for (int i = 0; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 96;
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
static void blit_to_target(UI* ui, uint8_t* target, int width, int height) {
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
  for (int i = 0; i < UI_HEIGHT; i++) {
    for (int j = 0; j < k; j++) {
      const int target_index = (i + ui->y) * 3 * width + 3 * ui->x + j * 48;
      const int ui_index     = j * 48 + i * UI_WIDTH * 3;

      __m128i ui1     = _mm_loadu_si128(ui->pixels + ui_index);
      __m128i mask1   = _mm_loadu_si128(ui->pixels_mask + ui_index);
      __m128i target1 = _mm_loadu_si128(target + target_index);
      target1         = _mm_avg_epu8(bg1, target1);
      target1         = _mm_blendv_epi8(target1, ui1, mask1);
      _mm256_storeu_si128(target + target_index, target1);
      __m128i ui2     = _mm_loadu_si128(ui->pixels + ui_index + 16);
      __m128i mask2   = _mm_loadu_si128(ui->pixels_mask + ui_index + 16);
      __m128i target2 = _mm_loadu_si128(target + target_index + 16);
      target2         = _mm_avg_epu8(bg2, target2);
      target2         = _mm_blendv_epi8(target2, ui2, mask2);
      _mm256_storeu_si128(target + target_index + 16, target2);
      __m128i ui3     = _mm_loadu_si128(ui->pixels + ui_index + 32);
      __m128i mask3   = _mm_loadu_si128(ui->pixels_mask + ui_index + 32);
      __m128i target3 = _mm_loadu_si128(target + target_index + 32);
      target3         = _mm_avg_epu8(bg3, target3);
      target3         = _mm_blendv_epi8(target3, ui3, mask3);
      _mm_storeu_si128(target + target_index + 32, target3);
    }
  }
}
#endif

void blit_UI(UI* ui, uint8_t* target, int width, int height) {
  blur_background(ui, target, width, height);
  blit_to_target(ui, target, width, height);
}

void free_UI(UI* ui) {
  free(ui->pixels);
  free(ui->pixels_mask);
  free(ui->scratch);

  for (int i = 0; i < UI_PANELS_GENERAL_COUNT; i++) {
    free_UIPanel(ui->general_panels + i);
  }

  free(ui->general_panels);
}
