#include "ui_renderer.h"

#include <assert.h>
#include <math.h>

#include "display.h"
#include "ui_renderer_blur.h"
#include "ui_renderer_impl.h"
#include "ui_renderer_utils.h"

#define R2_PHI1 3242174889u /* 0.7548776662f  */
#define R2_PHI2 2447445413u /* 0.56984029f    */

inline float _ui_renderer_rand(const uint32_t offset, const uint32_t phi) {
  const uint32_t v = offset * phi;

  const uint32_t i = 0x3F800000u | (v >> 9);

  return (*(float*) &i) - 1.0f;
}

////////////////////////////////////////////////////////////////////
// Mask creation
////////////////////////////////////////////////////////////////////

static void _ui_renderer_create_block_mask(uint8_t* dst_fill, uint8_t* dst_border, uint32_t size) {
  for (uint32_t y = 0; y < size; y++) {
    for (uint32_t x = 0; x < size; x++) {
      dst_fill[4 * (x + y * size) + 0] = 0xFF;
      dst_fill[4 * (x + y * size) + 1] = 0;
      dst_fill[4 * (x + y * size) + 2] = 0;
      dst_fill[4 * (x + y * size) + 3] = 0;

      dst_border[4 * (x + y * size) + 0] = (x == 0 || y == 0 || x + 1 == size || y + 1 == size) ? 0xFF : 0;
      dst_border[4 * (x + y * size) + 1] = 0;
      dst_border[4 * (x + y * size) + 2] = 0;
      dst_border[4 * (x + y * size) + 3] = 0;
    }
  }
}

static void _ui_renderer_create_disk_mask(uint8_t* dst, uint32_t size) {
  const float radius = size * 0.5f;

  const float radius_sq = radius * radius;

  for (uint32_t y = 0; y < size; y++) {
    for (uint32_t x = 0; x < size; x++) {
      uint32_t opacity = 0;
      for (uint32_t sample_id = 0; sample_id < 0xFF; sample_id++) {
        const float ox = x + _ui_renderer_rand(sample_id, R2_PHI1);
        const float oy = y + _ui_renderer_rand(sample_id, R2_PHI2);

        const float dx = fabsf(radius - ox);
        const float dy = fabsf(radius - oy);

        const bool is_inside = (dx * dx + dy * dy) <= radius_sq;

        opacity += is_inside ? 1 : 0;
      }

      dst[4 * (x + y * size) + 0] = opacity;
      dst[4 * (x + y * size) + 1] = 0;
      dst[4 * (x + y * size) + 2] = 0;
      dst[4 * (x + y * size) + 3] = 0;
    }
  }
}

static void _ui_renderer_create_circle_mask(uint8_t* dst, uint32_t size) {
  const float outer_radius = size * 0.5f;
  const float inner_radius = size * 0.5f - 1.0f;

  const float outer_radius_sq = outer_radius * outer_radius;
  const float inner_radius_sq = inner_radius * inner_radius;

  for (uint32_t y = 0; y < size; y++) {
    for (uint32_t x = 0; x < size; x++) {
      uint32_t opacity = 0;
      for (uint32_t sample_id = 0; sample_id < 0xFF; sample_id++) {
        const float ox = x + _ui_renderer_rand(sample_id, R2_PHI1);
        const float oy = y + _ui_renderer_rand(sample_id, R2_PHI2);

        const float dx = fabsf(outer_radius - ox);
        const float dy = fabsf(outer_radius - oy);

        const bool is_inside = (dx * dx + dy * dy) <= outer_radius_sq && (dx * dx + dy * dy) >= inner_radius_sq;

        opacity += is_inside ? 1 : 0;
      }

      dst[4 * (x + y * size) + 0] = opacity;
      dst[4 * (x + y * size) + 1] = 0;
      dst[4 * (x + y * size) + 2] = 0;
      dst[4 * (x + y * size) + 3] = 0;
    }
  }
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void ui_renderer_create(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);

  LUM_FAILURE_HANDLE(host_malloc(renderer, sizeof(UIRenderer)));

  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->block_mask, sizeof(uint32_t) * 2 * UI_RENDERER_STRIDE * 2 * UI_RENDERER_STRIDE));
  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->block_mask_border, sizeof(uint32_t) * 2 * UI_RENDERER_STRIDE * 2 * UI_RENDERER_STRIDE));
  (*renderer)->block_mask_size = 2 * UI_RENDERER_STRIDE;

  _ui_renderer_create_block_mask((*renderer)->block_mask, (*renderer)->block_mask_border, (*renderer)->block_mask_size);

  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = 16 << size_id;

    LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->disk_mask[size_id], sizeof(uint32_t) * size * size));
    LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->circle_mask[size_id], sizeof(uint32_t) * size * size));

    _ui_renderer_create_disk_mask((*renderer)->disk_mask[size_id], size);
    _ui_renderer_create_circle_mask((*renderer)->circle_mask[size_id], size);

    (*renderer)->shape_mask_size[size_id] = size;
  }
}

void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(window);

  ui_renderer_stack_blur(renderer, display, window);
}

void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(window);

  if (window->y >= (int32_t) display->height)
    return;

  const uint32_t rounding_size = (window->height >= 64) ? WINDOW_ROUNDING_SIZE : 0;
  const uint32_t height        = (window->y + window->height > display->height) ? display->height - window->y : window->height;

  _ui_renderer_render_rounded_box(
    renderer, window->width, window->height, window->background_blur_buffer, 0, 0, window->background_blur_buffer_ld, display->buffer,
    window->x, window->y, display->ld, rounding_size, height, 0, 0xFF111928, UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
}

void ui_renderer_render_rounded_box(
  UIRenderer* renderer, Display* display, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t rounding_size,
  uint32_t border_color, uint32_t background_color, UIRendererBackgroundMode background_mode) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);

  if (y >= display->height)
    return;

  const uint32_t height_clip = (y + height > display->height) ? display->height - y : height;

  _ui_renderer_render_rounded_box(
    renderer, width, height, display->buffer, x, y, display->ld, display->buffer, x, y, display->ld, rounding_size, height_clip,
    border_color, background_color, background_mode);
}

void ui_renderer_destroy(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(*renderer);

  LUM_FAILURE_HANDLE(host_free(&(*renderer)->block_mask));
  LUM_FAILURE_HANDLE(host_free(&(*renderer)->block_mask_border));

  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    LUM_FAILURE_HANDLE(host_free(&(*renderer)->disk_mask[size_id]));
    LUM_FAILURE_HANDLE(host_free(&(*renderer)->circle_mask[size_id]));
  }

  LUM_FAILURE_HANDLE(host_free(renderer));
}
