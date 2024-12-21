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

////////////////////////////////////////////////////////////////////
// Mask creation
////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////
// Render functions
////////////////////////////////////////////////////////////////////

static void _ui_renderer_upscale(const uint8_t* src, uint32_t width, uint32_t height, uint32_t src_ld, uint8_t* dst, uint32_t dst_ld) {
  const uint32_t cols = width >> (UI_RENDERER_STRIDE_LOG - 1);
  const uint32_t rows = height;

  const uint8_t* src_ptr = src;
  uint8_t* dst_ptr       = dst;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      Color128 base = color128_load(src_ptr + col * (UI_RENDERER_STRIDE_BYTES >> 1));

      Color256 upper = color256_load(dst_ptr + 0 + col * UI_RENDERER_STRIDE_BYTES);
      Color256 lower = color256_load(dst_ptr + dst_ld + col * UI_RENDERER_STRIDE_BYTES);

      Color256 color = color128_extend(base);
      color          = color256_shift_left64(color, 32);
      color          = color256_or(color, color256_shift_right64(color, 32));

      upper = color256_avg8(upper, color);
      lower = color256_avg8(lower, color);

      color256_store(dst_ptr + 0 + col * UI_RENDERER_STRIDE_BYTES, upper);
      color256_store(dst_ptr + dst_ld + col * UI_RENDERER_STRIDE_BYTES, lower);
    }

    src_ptr = src_ptr + src_ld;
    dst_ptr = dst_ptr + 2 * dst_ld;
  }
}

static void _ui_renderer_downscale(const uint8_t* src, uint32_t width, uint32_t height, uint32_t src_ld, uint8_t* dst, uint32_t dst_ld) {
  const uint32_t cols = width >> (UI_RENDERER_STRIDE_LOG + 1);
  const uint32_t rows = height >> 1;

  const uint8_t* src_ptr = src;
  uint8_t* dst_ptr       = dst;

  Color256 one = color256_set_1(0x01010101);

  Color256 shuffle_mask = color256_set(0x0F0B0E0A, 0x0D090C08, 0x07030602, 0x05010400, 0x0F0B0E0A, 0x0D090C08, 0x07030602, 0x05010400);

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      Color256 base00 = color256_load(src_ptr + 0 + col * 2 * UI_RENDERER_STRIDE_BYTES + 0);
      Color256 base01 = color256_load(src_ptr + 0 + col * 2 * UI_RENDERER_STRIDE_BYTES + UI_RENDERER_STRIDE_BYTES);

      Color256 base10 = color256_load(src_ptr + src_ld + col * 2 * UI_RENDERER_STRIDE_BYTES + 0);
      Color256 base11 = color256_load(src_ptr + src_ld + col * 2 * UI_RENDERER_STRIDE_BYTES + UI_RENDERER_STRIDE_BYTES);

      Color256 left  = color256_avg8(base00, base10);
      Color256 right = color256_avg8(base01, base11);

      left  = color256_shuffle8(left, shuffle_mask);
      right = color256_shuffle8(right, shuffle_mask);

      left  = color256_maddubs16(left, one);
      right = color256_maddubs16(right, one);

      left  = color256_shift_right16(left, 1);
      right = color256_shift_right16(right, 1);

      Color256 lower_bits  = color256_permute128(left, right, 0x20);
      Color256 higher_bits = color256_permute128(left, right, 0x31);

      Color256 result = color256_packus16(lower_bits, higher_bits);

      color256_store(dst_ptr + col * UI_RENDERER_STRIDE_BYTES, result);
    }

    src_ptr = src_ptr + 2 * src_ld;
    dst_ptr = dst_ptr + dst_ld;
  }
}

static void _ui_renderer_create_window_background(UIRenderer* renderer, Window* window, uint8_t* src, uint32_t ld) {
  uint8_t* ptr = src + window->y * ld + window->x * 4;

  _ui_renderer_downscale(ptr, window->width, window->height, ld, window->background_blur_buffers[0], window->background_blur_buffers_ld[0]);

  for (uint32_t mip_id = 0; mip_id + 1 < window->background_blur_mip_count; mip_id++) {
    const uint32_t shift_size = mip_id + 1;

    const uint32_t width  = window->width >> shift_size;
    const uint32_t height = window->height >> shift_size;

    const uint8_t* src_ptr = window->background_blur_buffers[mip_id];
    const uint32_t src_ld  = window->background_blur_buffers_ld[mip_id];

    uint8_t* dst_ptr      = window->background_blur_buffers[mip_id + 1];
    const uint32_t dst_ld = window->background_blur_buffers_ld[mip_id + 1];

    _ui_renderer_downscale(src_ptr, width, height, src_ld, dst_ptr, dst_ld);
  }

  for (uint32_t mip_id = window->background_blur_mip_count - 1; mip_id > 0; mip_id--) {
    const uint32_t shift_size = mip_id + 1;

    const uint32_t width  = window->width >> shift_size;
    const uint32_t height = window->height >> shift_size;

    const uint8_t* src_ptr = window->background_blur_buffers[mip_id];
    const uint32_t src_ld  = window->background_blur_buffers_ld[mip_id];

    uint8_t* dst_ptr      = window->background_blur_buffers[mip_id - 1];
    const uint32_t dst_ld = window->background_blur_buffers_ld[mip_id - 1];

    _ui_renderer_upscale(src_ptr, width, height, src_ld, dst_ptr, dst_ld);
  }

  _ui_renderer_upscale(
    window->background_blur_buffers[0], window->width >> 1, window->height >> 1, window->background_blur_buffers_ld[0], ptr, ld);
}

static void _ui_renderer_render_window(UIRenderer* renderer, Window* window, uint8_t* dst, uint32_t ld) {
  const uint32_t cols = window->width >> 3;
  const uint32_t rows = window->height;

  static_assert(UI_UNIT_SIZE == 16, "This was written with UI_UNIT_SIZE==16.");

  dst = dst + 4 * window->x + window->y * ld;

  uint32_t row = 0;

  Color256 base_color       = color256_set_1(0xFFD4AF37);
  Color256 background_color = color256_set_1(0xFF000000);
  Color256 mask_low16       = color256_set_1(0x00FF00FF);
  Color256 mask_high16      = color256_set_1(0xFF00FF00);
  Color256 mask_add         = color256_set_1(0x00800080);
  Color256 mask_full_alpha  = color256_set_1(0x000000FF);

  {
    Color256 circle_left = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 disk_left   = color256_load(renderer->disk_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 base        = color256_load(dst + 0 * 32);

    base = color256_alpha_blend(color256_avg8(base, background_color), base, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(dst + 0 * 32, color256_alpha_blend(base_color, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    color256_store(dst + col * 32, base_color);
  }

  {
    Color256 circle_right = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 disk_right   = color256_load(renderer->disk_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 base         = color256_load(dst + (cols - 1) * 32);

    base =
      color256_alpha_blend(color256_avg8(base, background_color), base, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(base_color, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  row++;
  dst = dst + ld;

  for (; row < 8; row++) {
    Color256 circle_left = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 disk_left   = color256_load(renderer->disk_mask + row * 4 * UI_UNIT_SIZE + 0);
    Color256 base_left   = color256_load(dst + 0 * 32);

    base_left = color256_alpha_blend(
      color256_avg8(base_left, background_color), base_left, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + 0 * 32, color256_alpha_blend(base_color, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    for (uint32_t col = 1; col < cols - 1; col++) {
      Color256 base = color256_load(dst + col * 32);
      base          = color256_avg8(base, background_color);

      color256_store(dst + col * 32, base);
    }

    Color256 circle_right = color256_load(renderer->circle_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 disk_right   = color256_load(renderer->disk_mask + row * 4 * UI_UNIT_SIZE + 32);
    Color256 base_right   = color256_load(dst + (cols - 1) * 32);

    base_right = color256_alpha_blend(
      color256_avg8(base_right, background_color), base_right, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + (cols - 1) * 32,
      color256_alpha_blend(base_color, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    dst = dst + ld;
  }

  {
    Color256 left_mask  = color256_set(0, 0, 0, 0, 0, 0, 0, 0xFF);
    Color256 right_mask = color256_set(0xFF, 0, 0, 0, 0, 0, 0, 0);

    for (; row < rows - 8; row++) {
      Color256 base_left = color256_load(dst + 0 * 32);
      base_left          = color256_avg8(base_left, background_color);

      color256_store(
        dst + 0 * 32, color256_alpha_blend(base_color, base_left, left_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));

      for (uint32_t col = 1; col < cols - 1; col++) {
        Color256 base = color256_load(dst + col * 32);
        base          = color256_avg8(base, background_color);

        color256_store(dst + col * 32, base);
      }

      Color256 base_right = color256_load(dst + (cols - 1) * 32);
      base_right          = color256_avg8(base_right, background_color);

      color256_store(
        dst + (cols - 1) * 32,
        color256_alpha_blend(base_color, base_right, right_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));

      dst = dst + ld;
    }
  }

  for (; row < rows - 1; row++) {
    Color256 circle_left = color256_load(renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 0);
    Color256 disk_left   = color256_load(renderer->disk_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 0);
    Color256 base_left   = color256_load(dst + 0 * 32);

    base_left = color256_alpha_blend(
      color256_avg8(base_left, background_color), base_left, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + 0 * 32, color256_alpha_blend(base_color, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    for (uint32_t col = 1; col < cols - 1; col++) {
      Color256 base = color256_load(dst + col * 32);
      base          = color256_avg8(base, background_color);

      color256_store(dst + col * 32, base);
    }

    Color256 circle_right = color256_load(renderer->circle_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 32);
    Color256 disk_right   = color256_load(renderer->disk_mask + (16 - (rows - row)) * 4 * UI_UNIT_SIZE + 32);
    Color256 base_right   = color256_load(dst + (cols - 1) * 32);

    base_right = color256_alpha_blend(
      color256_avg8(base_right, background_color), base_right, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + (cols - 1) * 32,
      color256_alpha_blend(base_color, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    dst = dst + ld;
  }

  {
    Color256 circle_left = color256_load(renderer->circle_mask + 15 * 4 * UI_UNIT_SIZE + 0);
    Color256 disk_left   = color256_load(renderer->disk_mask + 15 * 4 * UI_UNIT_SIZE + 0);
    Color256 base        = color256_load(dst + 0 * 32);

    base = color256_alpha_blend(color256_avg8(base, background_color), base, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(dst + 0 * 32, color256_alpha_blend(base_color, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }

  for (uint32_t col = 1; col < cols - 1; col++) {
    color256_store(dst + col * 32, base_color);
  }

  {
    Color256 circle_right = color256_load(renderer->circle_mask + 15 * 4 * UI_UNIT_SIZE + 32);
    Color256 disk_right   = color256_load(renderer->disk_mask + 15 * 4 * UI_UNIT_SIZE + 32);
    Color256 base         = color256_load(dst + (cols - 1) * 32);

    base =
      color256_alpha_blend(color256_avg8(base, background_color), base, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + (cols - 1) * 32, color256_alpha_blend(base_color, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));
  }
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void ui_renderer_create(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);

  LUM_FAILURE_HANDLE(host_malloc(renderer, sizeof(UIRenderer)));

  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->disk_mask, sizeof(uint32_t) * UI_UNIT_SIZE * UI_UNIT_SIZE));
  LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->circle_mask, sizeof(uint32_t) * UI_UNIT_SIZE * UI_UNIT_SIZE));

  _ui_renderer_create_disk_mask(*renderer);
  _ui_renderer_create_circle_mask(*renderer);
}

void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(window);

  _ui_renderer_create_window_background(renderer, window, display->buffer, display->ld);
}

void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);
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
