#include "ui_renderer_blur.h"

#include <stdlib.h>
#include <string.h>

#include "display.h"

#define BLUR_MAX_SUPPORTED_DIM (7680)

static void _ui_renderer_stack_blur_generic(
  const uint8_t* restrict src, uint32_t width, uint32_t height, uint32_t src_ld, uint32_t x, uint32_t y, uint32_t radius,
  uint8_t* restrict dst, uint32_t dst_ld) {
  const uint32_t cols = width;
  const uint32_t rows = height;

  if (radius > 254 || radius < 2) {
    error_message("Invalid blur radius. Skipping blur...");
    return;
  }

  if (rows > BLUR_MAX_SUPPORTED_DIM || cols > BLUR_MAX_SUPPORTED_DIM) {
    error_message("Blur not supported for given dimensions. Skipping blur...");
    return;
  }

  const size_t pixel_size = sizeof(LuminaryARGB8);

  /* Copy source rectangle into destination buffer (inplace target for results) */
  src = src + x * pixel_size + y * src_ld;

  size_t row_bytes = pixel_size * (size_t) cols;
  for (uint32_t row = 0; row < rows; row++)
    memcpy(dst + row * dst_ld, src + row * src_ld, row_bytes);

  uint8_t scratch_buffer[BLUR_MAX_SUPPORTED_DIM * sizeof(LuminaryARGB8)];

  uint32_t last_col = cols - 1;
  uint32_t last_row = rows - 1;

  const uint32_t L     = radius + 1; /* box length */
  const uint32_t left  = (L - 1) / 2;
  const uint32_t right = (L - 1) - left;

  /* Precompute normalization multiplier and shift to avoid divisions */
  const int NORM_SHIFT    = 20;
  const uint32_t norm_mul = (uint32_t) (((1ULL << NORM_SHIFT) + (L / 2)) / L);

  /* Horizontal two-pass (per-row). First pass: dst(row) -> row_tmp, second pass: row_tmp -> dst(row). */
  for (uint32_t row = 0; row < rows; ++row) {
    const uint8_t* restrict row_src = dst + row * dst_ld;
    uint8_t* restrict tmp_dst       = scratch_buffer;

    uint32_t sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    for (int k = -(int) left; k <= (int) right; ++k) {
      int pos = k;
      if (pos < 0)
        pos = 0;
      if ((uint32_t) pos > last_col)
        pos = (int) last_col;
      const uint8_t* restrict p = row_src + ((uint32_t) pos) * pixel_size;
      sum_r += p[0];
      sum_g += p[1];
      sum_b += p[2];
      sum_a += p[3];
    }

    for (uint32_t col = 0; col < cols; ++col) {
      uint32_t out_r = (uint32_t) ((sum_r * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_g = (uint32_t) ((sum_g * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_b = (uint32_t) ((sum_b * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_a = (uint32_t) ((sum_a * (uint64_t) norm_mul) >> NORM_SHIFT);

      uint8_t* restrict od = tmp_dst + col * pixel_size;

      od[0] = (uint8_t) out_r;
      od[1] = (uint8_t) out_g;
      od[2] = (uint8_t) out_b;
      od[3] = (uint8_t) out_a;

      int remove_idx = (int) col - (int) left;
      if (remove_idx < 0)
        remove_idx = 0;
      if ((uint32_t) remove_idx > last_col)
        remove_idx = (int) last_col;
      int add_idx = (int) col + (int) right + 1;
      if (add_idx < 0)
        add_idx = 0;
      if ((uint32_t) add_idx > last_col)
        add_idx = (int) last_col;

      const uint8_t* restrict p_remove = row_src + ((uint32_t) remove_idx) * pixel_size;
      const uint8_t* restrict p_add    = row_src + ((uint32_t) add_idx) * pixel_size;

      sum_r += (int) p_add[0] - (int) p_remove[0];
      sum_g += (int) p_add[1] - (int) p_remove[1];
      sum_b += (int) p_add[2] - (int) p_remove[2];
      sum_a += (int) p_add[3] - (int) p_remove[3];
    }

    /* second pass */
    uint32_t sum2_r = 0, sum2_g = 0, sum2_b = 0, sum2_a = 0;
    for (int k = -(int) left; k <= (int) right; ++k) {
      int pos = k;
      if (pos < 0)
        pos = 0;
      if ((uint32_t) pos > last_col)
        pos = (int) last_col;
      const uint8_t* restrict p = tmp_dst + ((uint32_t) pos) * pixel_size;
      sum2_r += p[0];
      sum2_g += p[1];
      sum2_b += p[2];
      sum2_a += p[3];
    }

    uint8_t* restrict row_dst = dst + row * dst_ld;
    for (uint32_t col = 0; col < cols; ++col) {
      uint32_t out_r = (uint32_t) ((sum2_r * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_g = (uint32_t) ((sum2_g * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_b = (uint32_t) ((sum2_b * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_a = (uint32_t) ((sum2_a * (uint64_t) norm_mul) >> NORM_SHIFT);

      uint8_t* restrict od = row_dst + col * pixel_size;

      od[0] = (uint8_t) out_r;
      od[1] = (uint8_t) out_g;
      od[2] = (uint8_t) out_b;
      od[3] = (uint8_t) out_a;

      int remove_idx = (int) col - (int) left;
      if (remove_idx < 0)
        remove_idx = 0;
      if ((uint32_t) remove_idx > last_col)
        remove_idx = (int) last_col;
      int add_idx = (int) col + (int) right + 1;
      if (add_idx < 0)
        add_idx = 0;
      if ((uint32_t) add_idx > last_col)
        add_idx = (int) last_col;

      const uint8_t* restrict p_remove = tmp_dst + ((uint32_t) remove_idx) * pixel_size;
      const uint8_t* restrict p_add    = tmp_dst + ((uint32_t) add_idx) * pixel_size;

      sum2_r += (int) p_add[0] - (int) p_remove[0];
      sum2_g += (int) p_add[1] - (int) p_remove[1];
      sum2_b += (int) p_add[2] - (int) p_remove[2];
      sum2_a += (int) p_add[3] - (int) p_remove[3];
    }
  }

  /* Vertical two-pass (per-column). First pass: dst(column) -> col_tmp, second pass: col_tmp -> dst(column). */
  for (uint32_t col = 0; col < cols; ++col) {
    uint32_t sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    for (int k = -(int) left; k <= (int) right; ++k) {
      int pos = k;
      if (pos < 0)
        pos = 0;
      if ((uint32_t) pos > last_row)
        pos = (int) last_row;
      const uint8_t* restrict p = dst + ((uint32_t) pos) * dst_ld + col * pixel_size;
      sum_r += p[0];
      sum_g += p[1];
      sum_b += p[2];
      sum_a += p[3];
    }

    for (uint32_t row = 0; row < rows; ++row) {
      uint8_t* restrict od = scratch_buffer + (size_t) row * pixel_size;

      uint32_t out_r = (uint32_t) ((sum_r * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_g = (uint32_t) ((sum_g * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_b = (uint32_t) ((sum_b * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_a = (uint32_t) ((sum_a * (uint64_t) norm_mul) >> NORM_SHIFT);

      od[0] = (uint8_t) out_r;
      od[1] = (uint8_t) out_g;
      od[2] = (uint8_t) out_b;
      od[3] = (uint8_t) out_a;

      int remove_idx = (int) row - (int) left;
      if (remove_idx < 0)
        remove_idx = 0;
      if ((uint32_t) remove_idx > last_row)
        remove_idx = (int) last_row;
      int add_idx = (int) row + (int) right + 1;
      if (add_idx < 0)
        add_idx = 0;
      if ((uint32_t) add_idx > last_row)
        add_idx = (int) last_row;

      const uint8_t* restrict p_remove = dst + ((uint32_t) remove_idx) * dst_ld + col * pixel_size;
      const uint8_t* restrict p_add    = dst + ((uint32_t) add_idx) * dst_ld + col * pixel_size;

      sum_r += (int) p_add[0] - (int) p_remove[0];
      sum_g += (int) p_add[1] - (int) p_remove[1];
      sum_b += (int) p_add[2] - (int) p_remove[2];
      sum_a += (int) p_add[3] - (int) p_remove[3];
    }

    /* second vertical pass: col_tmp -> dst column */
    uint32_t sum2_r = 0, sum2_g = 0, sum2_b = 0, sum2_a = 0;
    for (int k = -(int) left; k <= (int) right; ++k) {
      int pos = k;
      if (pos < 0)
        pos = 0;
      if ((uint32_t) pos > last_row)
        pos = (int) last_row;
      const uint8_t* restrict p = scratch_buffer + ((uint32_t) pos) * pixel_size;
      sum2_r += p[0];
      sum2_g += p[1];
      sum2_b += p[2];
      sum2_a += p[3];
    }

    for (uint32_t row = 0; row < rows; ++row) {
      uint8_t* restrict od = dst + (size_t) row * dst_ld + col * pixel_size;

      uint32_t out_r = (uint32_t) ((sum2_r * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_g = (uint32_t) ((sum2_g * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_b = (uint32_t) ((sum2_b * (uint64_t) norm_mul) >> NORM_SHIFT);
      uint32_t out_a = (uint32_t) ((sum2_a * (uint64_t) norm_mul) >> NORM_SHIFT);

      od[0] = (uint8_t) out_r;
      od[1] = (uint8_t) out_g;
      od[2] = (uint8_t) out_b;
      od[3] = (uint8_t) out_a;

      int remove_idx = (int) row - (int) left;
      if (remove_idx < 0)
        remove_idx = 0;
      if ((uint32_t) remove_idx > last_row)
        remove_idx = (int) last_row;
      int add_idx = (int) row + (int) right + 1;
      if (add_idx < 0)
        add_idx = 0;
      if ((uint32_t) add_idx > last_row)
        add_idx = (int) last_row;

      const uint8_t* restrict p_remove = scratch_buffer + ((uint32_t) remove_idx) * pixel_size;
      const uint8_t* restrict p_add    = scratch_buffer + ((uint32_t) add_idx) * pixel_size;

      sum2_r += (int) p_add[0] - (int) p_remove[0];
      sum2_g += (int) p_add[1] - (int) p_remove[1];
      sum2_b += (int) p_add[2] - (int) p_remove[2];
      sum2_a += (int) p_add[3] - (int) p_remove[3];
    }
  }
}

void ui_renderer_stack_blur(UIRenderer* renderer, Display* display, Window* window) {
  MD_UNUSED(renderer);

  if (window->y > (int32_t) display->height)
    return;

  const uint32_t height = (window->y + window->height > display->height) ? display->height - window->y : window->height;
  _ui_renderer_stack_blur_generic(
    display->buffer, window->width, height, display->pitch, window->x, window->y, 38, window->background_blur_buffer,
    window->background_blur_buffer_ld);
}
