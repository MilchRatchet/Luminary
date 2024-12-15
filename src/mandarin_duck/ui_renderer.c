#include "ui_renderer.h"

#include <math.h>

#include "display.h"
#include "ui_renderer_utils.h"

static void _ui_renderer_create_circle_mask(UIRenderer* renderer) {
  // TODO: This is just a test, do this properly later.
  for (uint32_t y = 0; y < UI_UNIT_SIZE; y++) {
    for (uint32_t x = 0; x < UI_UNIT_SIZE; x++) {
      const float dx = fabsf(8.0f - (x + 0.5f));
      const float dy = fabsf(8.0f - (y + 0.5f));

      const bool is_inside = (dx * dx + dy * dy) <= (8.0f * 8.0f);

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

  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->circle_mask, sizeof(uint32_t) * UI_UNIT_SIZE * UI_UNIT_SIZE));

  _ui_renderer_create_circle_mask(*renderer);
}

static void _ui_renderer_render_window(UIRenderer* renderer, Window* window, uint8_t* dst, uint32_t ld) {
  const uint32_t cols = window->width >> 3;
  const uint32_t rows = window->height;

  __m256i a = _mm256_set1_epi32(0x00000000);

  dst = dst + 4 * window->x + window->y * ld;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      _mm256_storeu_si256((__m256i*) (dst + col * 32), a);
    }

    dst = dst + ld;
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

  LUM_FAILURE_HANDLE(host_free(&(*renderer)->circle_mask));

  LUM_FAILURE_HANDLE(host_free(renderer));
}
