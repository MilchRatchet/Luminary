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

static void _ui_renderer_create_disk_mask(UIRenderer* renderer, uint8_t* dst, uint32_t size) {
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

static void _ui_renderer_create_circle_mask(UIRenderer* renderer, uint8_t* dst, uint32_t size) {
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

  uint32_t shape_mask_size = 32;

  uint32_t shape_mask_size_id = 0;
  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = renderer->shape_mask_size[size_id];

    if (size == shape_mask_size) {
      shape_mask_size_id = size_id;
      break;
    }
  }

  shape_mask_size = renderer->shape_mask_size[shape_mask_size_id];

  const uint8_t* disk_mask   = renderer->disk_mask[shape_mask_size_id];
  const uint8_t* circle_mask = renderer->circle_mask[shape_mask_size_id];

  const uint32_t shape_mask_half_size = shape_mask_size >> 1;
  const uint32_t shape_mask_ld        = shape_mask_size * sizeof(LuminaryARGB8);

  const uint32_t shape_mask_half_cols = shape_mask_half_size >> UI_RENDERER_STRIDE_LOG;

  dst = dst + sizeof(LuminaryARGB8) * window->x + window->y * ld;

  uint32_t row = 0;
  uint32_t col;

  uint32_t shape_row = 0;
  uint32_t shape_col;

  Color256 base_color       = color256_set_1(0xFFD4AF37);
  Color256 background_color = color256_set_1(0xFF0F0213);
  Color256 mask_low16       = color256_set_1(0x00FF00FF);
  Color256 mask_high16      = color256_set_1(0xFF00FF00);
  Color256 mask_add         = color256_set_1(0x00800080);
  Color256 mask_full_alpha  = color256_set_1(0x000000FF);

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  col       = 0;
  shape_col = 0;

  for (; col < shape_mask_half_cols; col++) {
    const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
    const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

    Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 disk_left   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 base        = color256_load(dst + col_offset);

    base = color256_alpha_blend(color256_avg8(base, background_color), base, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + col_offset, color256_alpha_blend(base_color, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    shape_col++;
  }

  for (; col < cols - shape_mask_half_cols; col++) {
    const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

    color256_store(dst + col_offset, base_color);
  }

  for (; col < cols; col++) {
    const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
    const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

    Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 disk_right   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 base         = color256_load(dst + col_offset);

    base =
      color256_alpha_blend(color256_avg8(base, background_color), base, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + col_offset, color256_alpha_blend(base_color, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    shape_col++;
  }

  row++;
  shape_row++;
  dst = dst + ld;

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  for (; row < shape_mask_half_size; row++) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 disk_left   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_left   = color256_load(dst + col_offset);

      base_left = color256_alpha_blend(
        color256_avg8(base_left, background_color), base_left, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      color256_store(
        dst + col_offset, color256_alpha_blend(base_color, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base = color256_load(dst + col_offset);
      base          = color256_avg8(base, background_color);

      color256_store(dst + col_offset, base);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 disk_right   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_right   = color256_load(dst + col_offset);

      base_right = color256_alpha_blend(
        color256_avg8(base_right, background_color), base_right, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      color256_store(
        dst + col_offset, color256_alpha_blend(base_color, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

      shape_col++;
    }

    shape_row++;
    dst = dst + ld;
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  {
    Color256 left_mask  = color256_set(0, 0, 0, 0, 0, 0, 0, 0xFF);
    Color256 right_mask = color256_set(0xFF, 0, 0, 0, 0, 0, 0, 0);

    for (; row < rows - shape_mask_half_size; row++) {
      uint32_t col = 0;

      for (; col < 1; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base_left = color256_load(dst + col_offset);
        base_left          = color256_avg8(base_left, background_color);

        color256_store(
          dst + col_offset, color256_alpha_blend(base_color, base_left, left_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));
      }

      for (; col < cols - 1; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base = color256_load(dst + col_offset);
        base          = color256_avg8(base, background_color);

        color256_store(dst + col_offset, base);
      }

      for (; col < cols; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base_right = color256_load(dst + col_offset);
        base_right          = color256_avg8(base_right, background_color);

        color256_store(
          dst + col_offset, color256_alpha_blend(base_color, base_right, right_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));
      }

      dst = dst + ld;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  for (; row < rows - 1; row++) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 disk_left   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_left   = color256_load(dst + col_offset);

      base_left = color256_alpha_blend(
        color256_avg8(base_left, background_color), base_left, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      color256_store(
        dst + col_offset, color256_alpha_blend(base_color, base_left, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base = color256_load(dst + col_offset);
      base          = color256_avg8(base, background_color);

      color256_store(dst + col_offset, base);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 disk_right   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_right   = color256_load(dst + col_offset);

      base_right = color256_alpha_blend(
        color256_avg8(base_right, background_color), base_right, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      color256_store(
        dst + col_offset, color256_alpha_blend(base_color, base_right, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

      shape_col++;
    }

    shape_row++;
    dst = dst + ld;
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  col       = 0;
  shape_col = 0;

  for (; col < shape_mask_half_cols; col++) {
    const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
    const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

    Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 disk_left   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 base        = color256_load(dst + col_offset);

    base = color256_alpha_blend(color256_avg8(base, background_color), base, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + col_offset, color256_alpha_blend(base_color, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha));

    shape_col++;
  }

  for (; col < cols - shape_mask_half_cols; col++) {
    const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

    color256_store(dst + col_offset, base_color);
  }

  for (; col < cols; col++) {
    const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
    const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

    Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 disk_right   = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
    Color256 base         = color256_load(dst + col_offset);

    base =
      color256_alpha_blend(color256_avg8(base, background_color), base, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

    color256_store(
      dst + col_offset, color256_alpha_blend(base_color, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha));

    shape_col++;
  }
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void ui_renderer_create(UIRenderer** renderer) {
  MD_CHECK_NULL_ARGUMENT(renderer);

  LUM_FAILURE_HANDLE(host_malloc(renderer, sizeof(UIRenderer)));

  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = 8 << size_id;

    LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->disk_mask[size_id], sizeof(uint32_t) * size * size));
    LUM_FAILURE_HANDLE(host_malloc(&(*renderer)->circle_mask[size_id], sizeof(uint32_t) * size * size));

    _ui_renderer_create_disk_mask(*renderer, (*renderer)->disk_mask[size_id], size);
    _ui_renderer_create_circle_mask(*renderer, (*renderer)->circle_mask[size_id], size);

    (*renderer)->shape_mask_size[size_id] = size;
  }
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

  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    LUM_FAILURE_HANDLE(host_free(&(*renderer)->disk_mask[size_id]));
    LUM_FAILURE_HANDLE(host_free(&(*renderer)->circle_mask[size_id]));
  }

  LUM_FAILURE_HANDLE(host_free(renderer));
}
