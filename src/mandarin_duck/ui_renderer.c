#include "ui_renderer.h"

#include <assert.h>
#include <math.h>

#include "display.h"
#include "ui_renderer_utils.h"

#define R2_PHI1 3242174889u /* 0.7548776662f  */
#define R2_PHI2 2447445413u /* 0.56984029f    */

inline float _ui_renderer_rand(const uint32_t offset, const uint32_t phi) {
  const uint32_t v = offset * phi;

  const uint32_t i = 0x3F800000u | (v >> 9);

  return (*(float*) &i) - 1.0f;
}

static void _ui_renderer_create_disk_mask(UIRenderer* renderer) {
  // TODO: This is just a test, do this properly later.
  for (uint32_t y = 0; y < UI_UNIT_SIZE; y++) {
    for (uint32_t x = 0; x < UI_UNIT_SIZE; x++) {
      uint32_t opacity = 0;
      for (uint32_t sample_id = 0; sample_id < 0xFF; sample_id++) {
        const float ox = x + _ui_renderer_rand(sample_id, R2_PHI1);
        const float oy = y + _ui_renderer_rand(sample_id, R2_PHI2);

        const float dx = fabsf(8.0f - ox);
        const float dy = fabsf(8.0f - oy);

        const bool is_inside = (dx * dx + dy * dy) <= (8.0f * 8.0f);

        opacity += is_inside ? 1 : 0;
      }

      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 0] = opacity;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 1] = 0;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 2] = 0;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 3] = 0;
    }
  }
}

static void _ui_renderer_create_circle_mask(UIRenderer* renderer) {
  // TODO: This is just a test, do this properly later.
  for (uint32_t y = 0; y < UI_UNIT_SIZE; y++) {
    for (uint32_t x = 0; x < UI_UNIT_SIZE; x++) {
      uint32_t opacity = 0;
      for (uint32_t sample_id = 0; sample_id < 0xFF; sample_id++) {
        const float ox = x + _ui_renderer_rand(sample_id, R2_PHI1);
        const float oy = y + _ui_renderer_rand(sample_id, R2_PHI2);

        const float dx = fabsf(8.0f - ox);
        const float dy = fabsf(8.0f - oy);

        const bool is_inside = (dx * dx + dy * dy) <= (8.0f * 8.0f) && (dx * dx + dy * dy) >= (7.0f * 7.0f);

        opacity += is_inside ? 1 : 0;
      }

      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 0] = opacity;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 1] = 0;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 2] = 0;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 3] = 0;
    }
  }
}

void ui_renderer_create(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);

  LUM_FAILURE_HANDLE(host_malloc(renderer, sizeof(UIRenderer)));

  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->disk_mask, sizeof(uint32_t) * UI_UNIT_SIZE * UI_UNIT_SIZE));
  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->circle_mask, sizeof(uint32_t) * UI_UNIT_SIZE * UI_UNIT_SIZE));

  _ui_renderer_create_disk_mask(*renderer);
  _ui_renderer_create_circle_mask(*renderer);
}

static void _ui_renderer_render_window(UIRenderer* renderer, Window* window, uint8_t* dst, uint32_t ld) {
  const uint32_t cols = window->width >> 3;
  const uint32_t rows = window->height;

  static_assert(UI_UNIT_SIZE == 16, "This was written with UI_UNIT_SIZE==16.");

  __m256i left  = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);
  __m256i right = _mm256_set_epi32(0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0);

  dst = dst + 4 * window->x + window->y * ld;

  uint32_t row = 0;

  Color256 white           = color256_set_1(0xFFFFFFFF);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  {
    Color256 circle_left = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 base        = color256_load(dst + 0 * 32);

    color256_store(dst + 0 * 32, color256_alpha_blend(white, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    color256_store(dst + col * 32, white);
  }

  {
    Color256 circle_right = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 base         = color256_load(dst + (cols - 1) * 32);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(white, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  row++;
  dst = dst + ld;

  for (; row < 8; row++) {
    Color256 circle_left = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 base_left   = color256_load(dst + 0 * 32);

    color256_store(dst + 0 * 32, color256_alpha_blend(white, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    Color256 circle_right = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 base_right   = color256_load(dst + (cols - 1) * 32);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(white, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    dst = dst + ld;
  }

  for (; row < rows - 8; row++) {
    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), left);
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), right);

    dst = dst + ld;
  }

  for (; row < rows - 1; row++) {
    Color256 circle_left = color256_load(renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 0);
    Color256 base_left   = color256_load(dst + 0 * 32);

    color256_store(dst + 0 * 32, color256_alpha_blend(white, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    Color256 circle_right = color256_load(renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 32);
    Color256 base_right   = color256_load(dst + (cols - 1) * 32);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(white, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    dst = dst + ld;
  }

  {
    Color256 circle_left = color256_load(renderer->circle_mask + 15 * 4 * UI_UNIT_SIZE + 0);
    Color256 base        = color256_load(dst + 0 * 32);

    color256_store(dst + 0 * 32, color256_alpha_blend(white, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    color256_store(dst + col * 32, white);
  }

  {
    Color256 circle_right = color256_load(renderer->circle_mask + 15 * 4 * UI_UNIT_SIZE + 32);
    Color256 base         = color256_load(dst + (cols - 1) * 32);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(white, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }
}

void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(window);

  _ui_renderer_render_window(renderer, window, display->buffer, display->ld);
}

void ui_renderer_destroy(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(*renderer);

  LUM_FAILURE_HANDLE(host_free(&(*renderer)->disk_mask));
  LUM_FAILURE_HANDLE(host_free(&(*renderer)->circle_mask));

  LUM_FAILURE_HANDLE(host_free(renderer));
}
