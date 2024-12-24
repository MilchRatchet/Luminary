#include "ui_renderer_blur.h"

#include "display.h"

/*
 * Based on the implementation in https://github.com/FigBug/Gin
 */

static uint16_t const stackblur_mul[255] = {
  512, 512, 456, 512, 328, 456, 335, 512, 405, 328, 271, 456, 388, 335, 292, 512, 454, 405, 364, 328, 298, 271, 496, 456, 420, 388,
  360, 335, 312, 292, 273, 512, 482, 454, 428, 405, 383, 364, 345, 328, 312, 298, 284, 271, 259, 496, 475, 456, 437, 420, 404, 388,
  374, 360, 347, 335, 323, 312, 302, 292, 282, 273, 265, 512, 497, 482, 468, 454, 441, 428, 417, 405, 394, 383, 373, 364, 354, 345,
  337, 328, 320, 312, 305, 298, 291, 284, 278, 271, 265, 259, 507, 496, 485, 475, 465, 456, 446, 437, 428, 420, 412, 404, 396, 388,
  381, 374, 367, 360, 354, 347, 341, 335, 329, 323, 318, 312, 307, 302, 297, 292, 287, 282, 278, 273, 269, 265, 261, 512, 505, 497,
  489, 482, 475, 468, 461, 454, 447, 441, 435, 428, 422, 417, 411, 405, 399, 394, 389, 383, 378, 373, 368, 364, 359, 354, 350, 345,
  341, 337, 332, 328, 324, 320, 316, 312, 309, 305, 301, 298, 294, 291, 287, 284, 281, 278, 274, 271, 268, 265, 262, 259, 257, 507,
  501, 496, 491, 485, 480, 475, 470, 465, 460, 456, 451, 446, 442, 437, 433, 428, 424, 420, 416, 412, 408, 404, 400, 396, 392, 388,
  385, 381, 377, 374, 370, 367, 363, 360, 357, 354, 350, 347, 344, 341, 338, 335, 332, 329, 326, 323, 320, 318, 315, 312, 310, 307,
  304, 302, 299, 297, 294, 292, 289, 287, 285, 282, 280, 278, 275, 273, 271, 269, 267, 265, 263, 261, 259};

static uint8_t const stackblur_shr[255] = {
  9,  11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
  19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
  22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
  23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
  23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};

static void _ui_renderer_stack_blur_generic(
  const uint8_t* src, uint32_t width, uint32_t height, uint32_t src_ld, uint32_t x, uint32_t y, uint32_t radius, uint8_t* dst,
  uint32_t dst_ld) {
  const uint32_t cols = width;
  const uint32_t rows = height;

  if (radius > 254 || radius < 2) {
    crash_message("Invalid blur radius");
  }

  // TODO: We don't need this to be inplace, see if I can improve performance by using the property.
  src = src + x * sizeof(LuminaryARGB8) + y * src_ld;
  for (uint32_t row = 0; row < rows; row++) {
    memcpy(dst + row * dst_ld, src + row * src_ld, sizeof(LuminaryARGB8) * width);
  }

  uint8_t stack[(254 * 2 + 1) * 4];

  uint32_t xp, yp, sp, stack_start;

  uint8_t* stack_ptr = (uint8_t*) 0;

  uint64_t sum_r, sum_g, sum_b, sum_a, sum_in_r, sum_in_g, sum_in_b, sum_in_a, sum_out_r, sum_out_g, sum_out_b, sum_out_a;

  uint32_t last_col = cols - 1;
  uint32_t last_row = rows - 1;
  uint32_t div      = (radius * 2) + 1;
  uint32_t mul_sum  = stackblur_mul[radius];
  uint8_t shr_sum   = stackblur_shr[radius];

  for (uint32_t row = 0; row < rows; row++) {
    sum_r = sum_g = sum_b = sum_a = sum_in_r = sum_in_g = sum_in_b = sum_in_a = sum_out_r = sum_out_g = sum_out_b = sum_out_a = 0;

    const uint8_t* src_ptr = dst + row * dst_ld;

    for (uint32_t i = 0; i <= radius; i++) {
      stack_ptr    = &stack[4 * i];
      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];
      sum_r += src_ptr[0] * (i + 1);
      sum_g += src_ptr[1] * (i + 1);
      sum_b += src_ptr[2] * (i + 1);
      sum_a += src_ptr[3] * (i + 1);
      sum_out_r += src_ptr[0];
      sum_out_g += src_ptr[1];
      sum_out_b += src_ptr[2];
      sum_out_a += src_ptr[3];
    }

    for (uint32_t i = 1; i <= radius; ++i) {
      if (i <= last_col)
        src_ptr += 4;

      stack_ptr    = &stack[4 * (i + radius)];
      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];
      sum_r += src_ptr[0] * (radius + 1 - i);
      sum_g += src_ptr[1] * (radius + 1 - i);
      sum_b += src_ptr[2] * (radius + 1 - i);
      sum_a += src_ptr[3] * (radius + 1 - i);
      sum_in_r += src_ptr[0];
      sum_in_g += src_ptr[1];
      sum_in_b += src_ptr[2];
      sum_in_a += src_ptr[3];
    }

    sp = radius;
    xp = radius;
    if (xp > last_col)
      xp = last_col;

    src_ptr          = dst + row * dst_ld + xp * sizeof(LuminaryARGB8);
    uint8_t* dst_ptr = dst + row * dst_ld;

    for (uint32_t col = 0; col < cols; col++) {
      dst_ptr[0] = (uint8_t) ((sum_r * mul_sum) >> shr_sum);
      dst_ptr[1] = (uint8_t) ((sum_g * mul_sum) >> shr_sum);
      dst_ptr[2] = (uint8_t) ((sum_b * mul_sum) >> shr_sum);
      dst_ptr[3] = (uint8_t) ((sum_a * mul_sum) >> shr_sum);
      dst_ptr += 4;

      sum_r -= sum_out_r;
      sum_g -= sum_out_g;
      sum_b -= sum_out_b;
      sum_a -= sum_out_a;

      stack_start = sp + div - radius;

      if (stack_start >= div)
        stack_start -= div;

      stack_ptr = &stack[4 * stack_start];

      sum_out_r -= stack_ptr[0];
      sum_out_g -= stack_ptr[1];
      sum_out_b -= stack_ptr[2];
      sum_out_a -= stack_ptr[3];

      if (xp < last_col) {
        src_ptr += 4;
        ++xp;
      }

      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];

      sum_in_r += src_ptr[0];
      sum_in_g += src_ptr[1];
      sum_in_b += src_ptr[2];
      sum_in_a += src_ptr[3];
      sum_r += sum_in_r;
      sum_g += sum_in_g;
      sum_b += sum_in_b;
      sum_a += sum_in_a;

      ++sp;
      if (sp >= div)
        sp = 0;

      stack_ptr = &stack[sp * 4];

      sum_out_r += stack_ptr[0];
      sum_out_g += stack_ptr[1];
      sum_out_b += stack_ptr[2];
      sum_out_a += stack_ptr[3];
      sum_in_r -= stack_ptr[0];
      sum_in_g -= stack_ptr[1];
      sum_in_b -= stack_ptr[2];
      sum_in_a -= stack_ptr[3];
    }
  }

  for (uint32_t col = 0; col < cols; col++) {
    sum_r = sum_g = sum_b = sum_a = sum_in_r = sum_in_g = sum_in_b = sum_in_a = sum_out_r = sum_out_g = sum_out_b = sum_out_a = 0;

    const uint8_t* src_ptr = dst + col * sizeof(LuminaryARGB8);

    for (uint32_t i = 0; i <= radius; i++) {
      stack_ptr    = &stack[i * 4];
      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];
      sum_r += src_ptr[0] * (i + 1);
      sum_g += src_ptr[1] * (i + 1);
      sum_b += src_ptr[2] * (i + 1);
      sum_a += src_ptr[3] * (i + 1);
      sum_out_r += src_ptr[0];
      sum_out_g += src_ptr[1];
      sum_out_b += src_ptr[2];
      sum_out_a += src_ptr[3];
    }

    for (uint32_t i = 1; i <= radius; i++) {
      if (i <= last_row)
        src_ptr += dst_ld;

      stack_ptr    = &stack[4 * (i + radius)];
      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];
      sum_r += src_ptr[0] * (radius + 1 - i);
      sum_g += src_ptr[1] * (radius + 1 - i);
      sum_b += src_ptr[2] * (radius + 1 - i);
      sum_a += src_ptr[3] * (radius + 1 - i);
      sum_in_r += src_ptr[0];
      sum_in_g += src_ptr[1];
      sum_in_b += src_ptr[2];
      sum_in_a += src_ptr[3];
    }

    sp = radius;
    yp = radius;
    if (yp > last_row)
      yp = last_row;

    src_ptr          = dst + yp * dst_ld + col * sizeof(LuminaryARGB8);
    uint8_t* dst_ptr = dst + col * sizeof(LuminaryARGB8);

    for (uint32_t row = 0; row < rows; row++) {
      dst_ptr[0] = (uint8_t) ((sum_r * mul_sum) >> shr_sum);
      dst_ptr[1] = (uint8_t) ((sum_g * mul_sum) >> shr_sum);
      dst_ptr[2] = (uint8_t) ((sum_b * mul_sum) >> shr_sum);
      dst_ptr[3] = (uint8_t) ((sum_a * mul_sum) >> shr_sum);
      dst_ptr += dst_ld;

      sum_r -= sum_out_r;
      sum_g -= sum_out_g;
      sum_b -= sum_out_b;
      sum_a -= sum_out_a;

      stack_start = sp + div - radius;
      if (stack_start >= div)
        stack_start -= div;

      stack_ptr = &stack[4 * stack_start];

      sum_out_r -= stack_ptr[0];
      sum_out_g -= stack_ptr[1];
      sum_out_b -= stack_ptr[2];
      sum_out_a -= stack_ptr[3];

      if (yp < last_row) {
        src_ptr += dst_ld;
        ++yp;
      }

      stack_ptr[0] = src_ptr[0];
      stack_ptr[1] = src_ptr[1];
      stack_ptr[2] = src_ptr[2];
      stack_ptr[3] = src_ptr[3];

      sum_in_r += src_ptr[0];
      sum_in_g += src_ptr[1];
      sum_in_b += src_ptr[2];
      sum_in_a += src_ptr[3];
      sum_r += sum_in_r;
      sum_g += sum_in_g;
      sum_b += sum_in_b;
      sum_a += sum_in_a;

      ++sp;
      if (sp >= div)
        sp = 0;

      stack_ptr = &stack[sp * 4];

      sum_out_r += stack_ptr[0];
      sum_out_g += stack_ptr[1];
      sum_out_b += stack_ptr[2];
      sum_out_a += stack_ptr[3];
      sum_in_r -= stack_ptr[0];
      sum_in_g -= stack_ptr[1];
      sum_in_b -= stack_ptr[2];
      sum_in_a -= stack_ptr[3];
    }
  }
}

void ui_renderer_stack_blur(UIRenderer* renderer, Display* display, Window* window) {
  const uint32_t height = (window->y + window->height > display->height) ? display->height - window->y : window->height;
  _ui_renderer_stack_blur_generic(
    display->buffer, window->width, height, display->ld, window->x, window->y, 38, window->background_blur_buffer,
    window->background_blur_buffer_ld);
}
