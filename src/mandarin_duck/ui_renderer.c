#include "ui_renderer.h"

#include <assert.h>
#include <math.h>

#include "display.h"
#include "ui_renderer_blur.h"
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
// Render functions
////////////////////////////////////////////////////////////////////

static void _ui_renderer_render_rounded_box(
  UIRenderer* renderer, uint32_t width, uint32_t height, const uint8_t* src, uint32_t src_x, uint32_t src_y, uint32_t src_ld, uint8_t* dst,
  uint32_t dst_x, uint32_t dst_y, uint32_t dst_ld, uint32_t rounding_size, uint32_t height_clip, uint32_t border_color,
  uint32_t background_color) {
  const uint32_t cols = width >> UI_RENDERER_STRIDE_LOG;
  const uint32_t rows = height;

  uint32_t shape_mask_size = rounding_size;

  uint32_t shape_mask_size_id = 0xFFFFFFFF;
  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = renderer->shape_mask_size[size_id];

    if (size == shape_mask_size) {
      shape_mask_size_id = size_id;
      break;
    }
  }

  shape_mask_size = (shape_mask_size_id != 0xFFFFFFFF) ? renderer->shape_mask_size[shape_mask_size_id] : renderer->block_mask_size;

  const uint8_t* disk_mask   = (shape_mask_size_id != 0xFFFFFFFF) ? renderer->disk_mask[shape_mask_size_id] : renderer->block_mask;
  const uint8_t* circle_mask = (shape_mask_size_id != 0xFFFFFFFF) ? renderer->circle_mask[shape_mask_size_id] : renderer->block_mask_border;

  const uint32_t shape_mask_half_size = shape_mask_size >> 1;
  const uint32_t shape_mask_ld        = shape_mask_size * sizeof(LuminaryARGB8);

  const uint32_t shape_mask_half_cols = shape_mask_half_size >> UI_RENDERER_STRIDE_LOG;

  dst = dst + sizeof(LuminaryARGB8) * dst_x + dst_y * dst_ld;
  src = src + sizeof(LuminaryARGB8) * src_x + src_y * src_ld;

  uint32_t row = 0;
  uint32_t col;

  uint32_t shape_row = 0;
  uint32_t shape_col;

  Color256 border          = color256_set_1(border_color);
  Color256 background      = color256_set_1(background_color);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  const bool use_background = (background_color != 0);
  const bool use_border     = (border_color != 0);

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  if (row < height_clip) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_left = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst  = color256_load(dst + col_offset);
      Color256 base_src  = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                 = color256_alpha_blend(border, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base;
      if (use_border) {
        base = border;
      }
      else {
        base = color256_load(src + col_offset);

        if (use_background) {
          base = color256_avg8(base, background);
        }
      }

      color256_store(dst + col_offset, base);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_right = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst   = color256_load(dst + col_offset);
      Color256 base_src   = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                  = color256_alpha_blend(border, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    row++;
    shape_row++;
    dst = dst + dst_ld;
    src = src + src_ld;
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  for (; row < shape_mask_half_size && row < height_clip; row++) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_left = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst  = color256_load(dst + col_offset);
      Color256 base_src  = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                 = color256_alpha_blend(border, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base_src = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      color256_store(dst + col_offset, base_src);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_right = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst   = color256_load(dst + col_offset);
      Color256 base_src   = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                  = color256_alpha_blend(border, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    shape_row++;
    dst = dst + dst_ld;
    src = src + src_ld;
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  {
    Color256 left_mask  = color256_set(0, 0, 0, 0, 0, 0, 0, 0xFF);
    Color256 right_mask = color256_set(0xFF, 0, 0, 0, 0, 0, 0, 0);

    for (; row < rows - shape_mask_half_size && row < height_clip; row++) {
      uint32_t col = 0;

      for (; col < 1; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base_src = color256_load(src + col_offset);

        if (use_background) {
          base_src = color256_avg8(base_src, background);
        }

        if (use_border) {
          base_src = color256_alpha_blend(border, base_src, left_mask, mask_low16, mask_high16, mask_add, mask_full_alpha);
        }

        color256_store(dst + col_offset, base_src);
      }

      for (; col < cols - 1; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base_src = color256_load(src + col_offset);

        if (use_background) {
          base_src = color256_avg8(base_src, background);
        }

        color256_store(dst + col_offset, base_src);
      }

      for (; col < cols; col++) {
        const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

        Color256 base_src = color256_load(src + col_offset);

        if (use_background) {
          base_src = color256_avg8(base_src, background);
        }

        if (use_border) {
          base_src = color256_alpha_blend(border, base_src, right_mask, mask_low16, mask_high16, mask_add, mask_full_alpha);
        }

        color256_store(dst + col_offset, base_src);
      }

      dst = dst + dst_ld;
      src = src + src_ld;
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  for (; row < rows - 1 && row < height_clip; row++) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_left = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst  = color256_load(dst + col_offset);
      Color256 base_src  = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                 = color256_alpha_blend(border, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base_src = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      color256_store(dst + col_offset, base_src);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_right = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst   = color256_load(dst + col_offset);
      Color256 base_src   = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                  = color256_alpha_blend(border, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    shape_row++;
    dst = dst + dst_ld;
    src = src + src_ld;
  }

  ////////////////////////////////////////////////////////////////////
  // Row kind
  ////////////////////////////////////////////////////////////////////

  if (row < height_clip) {
    col       = 0;
    shape_col = 0;

    for (; col < shape_mask_half_cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_left = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst  = color256_load(dst + col_offset);
      Color256 base_src  = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_left = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                 = color256_alpha_blend(border, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
    }

    for (; col < cols - shape_mask_half_cols; col++) {
      const uint32_t col_offset = col * UI_RENDERER_STRIDE_BYTES;

      Color256 base;
      if (use_border) {
        base = border;
      }
      else {
        base = color256_load(src + col_offset);

        if (use_background) {
          base = color256_avg8(base, background);
        }
      }

      color256_store(dst + col_offset, base);
    }

    for (; col < cols; col++) {
      const uint32_t col_offset       = col * UI_RENDERER_STRIDE_BYTES;
      const uint32_t shape_col_offset = shape_col * UI_RENDERER_STRIDE_BYTES;

      Color256 disk_right = color256_load(disk_mask + shape_row * shape_mask_ld + shape_col_offset);
      Color256 base_dst   = color256_load(dst + col_offset);
      Color256 base_src   = color256_load(src + col_offset);

      if (use_background) {
        base_src = color256_avg8(base_src, background);
      }

      Color256 base = color256_alpha_blend(base_src, base_dst, disk_right, mask_low16, mask_high16, mask_add, mask_full_alpha);

      if (use_border) {
        Color256 circle_right = color256_load(circle_mask + shape_row * shape_mask_ld + shape_col_offset);
        base                  = color256_alpha_blend(border, base, circle_right, mask_low16, mask_high16, mask_add, mask_full_alpha);
      }

      color256_store(dst + col_offset, base);

      shape_col++;
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

  const uint32_t rounding_size = (window->height >= 64) ? 32 : 0;
  const uint32_t height        = (window->y + window->height > display->height) ? display->height - window->y : window->height;

  _ui_renderer_render_rounded_box(
    renderer, window->width, window->height, window->background_blur_buffer, 0, 0, window->background_blur_buffer_ld, display->buffer,
    window->x, window->y, display->ld, rounding_size, height, 0, 0xFF111928);
}

void ui_renderer_render_rounded_box(
  UIRenderer* renderer, Display* display, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t rounding_size) {
  MD_CHECK_NULL_ARGUMENT(renderer);
  MD_CHECK_NULL_ARGUMENT(display);

  if (y >= display->height)
    return;

  const uint32_t height_clip = (y + height > display->height) ? display->height - y : height;

  _ui_renderer_render_rounded_box(
    renderer, width, height, display->buffer, x, y, display->ld, display->buffer, x, y, display->ld, rounding_size, height_clip, 0xFF111111,
    0xFF000000);
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
