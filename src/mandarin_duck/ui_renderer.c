#include "ui_renderer.h"

#include <assert.h>
#include <math.h>

#include "display.h"
#include "ui_renderer_utils.h"

static void _ui_renderer_create_disk_mask(UIRenderer* renderer) {
  // TODO: This is just a test, do this properly later.
  for (uint32_t y = 0; y < UI_UNIT_SIZE; y++) {
    for (uint32_t x = 0; x < UI_UNIT_SIZE; x++) {
      const float dx = fabsf(8.0f - (x + 0.5f));
      const float dy = fabsf(8.0f - (y + 0.5f));

      const bool is_inside = (dx * dx + dy * dy) <= (8.0f * 8.0f);

      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 0] = is_inside ? 0xFF : 0;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 1] = is_inside ? 0xFF : 0;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 2] = is_inside ? 0xFF : 0;
      renderer->disk_mask[4 * (x + y * UI_UNIT_SIZE) + 3] = is_inside ? 0xFF : 0;
    }
  }
}

static void _ui_renderer_create_circle_mask(UIRenderer* renderer) {
  // TODO: This is just a test, do this properly later.
  for (uint32_t y = 0; y < UI_UNIT_SIZE; y++) {
    for (uint32_t x = 0; x < UI_UNIT_SIZE; x++) {
      const float dx = fabsf(8.0f - (x + 0.5f));
      const float dy = fabsf(8.0f - (y + 0.5f));

      const bool is_inside = (dx * dx + dy * dy) <= (8.0f * 8.0f) && (dx * dx + dy * dy) >= (7.0f * 7.0f);

      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 0] = is_inside ? 0xFF : 0;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 1] = is_inside ? 0xFF : 0;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 2] = is_inside ? 0xFF : 0;
      renderer->circle_mask[4 * (x + y * UI_UNIT_SIZE) + 3] = is_inside ? 0xFF : 0;
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

  __m256i white = _mm256_set1_epi32(0xFFFFFFFF);
  __m256i left  = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);
  __m256i right = _mm256_set_epi32(0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0);

  dst = dst + 4 * window->x + window->y * ld;

  uint32_t row = 0;

  {
    __m256i circle_left = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0));
    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), circle_left);
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    _mm256_storeu_si256((__m256i*) (dst + col * 32), white);
  }

  {
    __m256i circle_right = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32));
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), circle_right);
  }

  row++;
  dst = dst + ld;

  for (; row < 8; row++) {
    __m256i circle_left  = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0));
    __m256i circle_right = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32));

    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), circle_left);
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), circle_right);

    dst = dst + ld;
  }

  for (; row < rows - 8; row++) {
    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), left);
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), right);

    dst = dst + ld;
  }

  for (; row < rows - 1; row++) {
    __m256i circle_left  = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 0));
    __m256i circle_right = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 32));

    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), circle_left);
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), circle_right);

    dst = dst + ld;
  }

  {
    __m256i circle_left = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 0));
    _mm256_storeu_si256((__m256i*) (dst + 0 * 32), circle_left);
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    _mm256_storeu_si256((__m256i*) (dst + col * 32), white);
  }

  {
    __m256i circle_right = _mm256_loadu_si256((__m256i*) (renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 32));
    _mm256_storeu_si256((__m256i*) (dst + (cols - 1) * 32), circle_right);
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
