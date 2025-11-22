#ifndef CU_POST_COMMON_H
#define CU_POST_COMMON_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

LUMINARY_FUNCTION float post_sample_buffer(
  const float* buffer, const float x, const float y, const uint32_t width, const uint32_t height, const float mem_scale = 1.0f) {
  const float source_x = fmaxf(0.0f, x * (width - 1)) * mem_scale;
  const float source_y = fmaxf(0.0f, y * (height - 1)) * mem_scale;

  const uint32_t index_x0 = (uint32_t) source_x;
  const uint32_t index_y0 = (uint32_t) source_y;
  const uint32_t index_x1 = min((uint32_t) (source_x + mem_scale), width - 1);
  const uint32_t index_y1 = min((uint32_t) (source_y + mem_scale), height - 1);

  const uint32_t index_00 = index_x0 + index_y0 * width * mem_scale;
  const uint32_t index_01 = index_x0 + index_y1 * width * mem_scale;
  const uint32_t index_10 = index_x1 + index_y0 * width * mem_scale;
  const uint32_t index_11 = index_x1 + index_y1 * width * mem_scale;

  const float pixel_00 = __ldca(buffer + index_00);
  const float pixel_01 = __ldca(buffer + index_01);
  const float pixel_10 = __ldca(buffer + index_10);
  const float pixel_11 = __ldca(buffer + index_11);

  const float fx  = source_x - index_x0;
  const float ifx = 1.0f - fx;
  const float fy  = source_y - index_y0;
  const float ify = 1.0f - fy;

  const float f00 = ifx * ify;
  const float f01 = ifx * fy;
  const float f10 = fx * ify;
  const float f11 = fx * fy;

  float result = pixel_00 * f00;
  result += pixel_01 * f01;
  result += pixel_10 * f10;
  result += pixel_11 * f11;

  return result;
}

LUMINARY_FUNCTION float post_sample_buffer_clamp(
  const float* image, float x, float y, const int width, const int height, const float mem_scale = 1.0f) {
  x = fmaxf(x, 0.0f);
  y = fmaxf(y, 0.0f);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

  return post_sample_buffer(image, x, y, width, height, mem_scale);
}

LUMINARY_FUNCTION float post_sample_buffer_border(
  const float* image, float x, float y, const int width, const int height, const float weight = 1.0f) {
  if (x > __uint_as_float(0b00111111011111111111111111111111) || x < 0.0f) {
    return 0.0f;
  }

  if (y > __uint_as_float(0b00111111011111111111111111111111) || y < 0.0f) {
    return 0.0f;
  }

  return post_sample_buffer(image, x, y, width, height) * weight;
}

LUMINARY_KERNEL void post_image_downsample(const KernelArgsPostImageDownsample args) {
  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);
  const float step_x  = 1.0f / (args.sw - 1);
  const float step_y  = 1.0f / (args.sh - 1);

  const uint32_t amount = args.tw * args.th;

  for (uint32_t index = THREAD_ID; index < amount; index += NUM_THREADS) {
    const uint32_t y = index / args.tw;
    const uint32_t x = index - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    float pixel = 0.0f;

    pixel += post_sample_buffer_border(args.src, sx - 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx + 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx - 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx + 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);

    pixel += post_sample_buffer_border(args.src, sx, sy, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx, sy - step_y, args.sw, args.sh, 0.5f);
    pixel += post_sample_buffer_border(args.src, sx - step_x, sy, args.sw, args.sh, 0.5f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy, args.sw, args.sh, 0.5f);
    pixel += post_sample_buffer_border(args.src, sx, sy + step_y, args.sw, args.sh, 0.5f);
    pixel += post_sample_buffer_border(args.src, sx - step_x, sy - step_y, args.sw, args.sh, 0.25f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy - step_y, args.sw, args.sh, 0.25f);
    pixel += post_sample_buffer_border(args.src, sx - step_x, sy + step_y, args.sw, args.sh, 0.25f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy + step_y, args.sw, args.sh, 0.25f);

    pixel *= 1.0f / 8.0f;

    pixel = fmaxf(pixel - args.threshold, 0.0f);

    __stcs(args.dst + x + y * args.tw, pixel);
  }
}

LUMINARY_KERNEL void post_image_upsample(const KernelArgsPostImageUpsample args) {
  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);
  const float step_x  = 1.0f / (args.sw - 1);
  const float step_y  = 1.0f / (args.sh - 1);

  const uint32_t amount = args.tw * args.th;

  for (uint32_t index = THREAD_ID; index < amount; index += NUM_THREADS) {
    const uint32_t y = index / args.tw;
    const uint32_t x = index - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    float pixel = post_sample_buffer_border(args.src, sx - step_x, sy - step_y, args.sw, args.sh);

    pixel += post_sample_buffer_border(args.src, sx, sy - step_y, args.sw, args.sh, 2.0f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy - step_y, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx - step_x, sy, args.sw, args.sh, 2.0f);
    pixel += post_sample_buffer_border(args.src, sx, sy, args.sw, args.sh, 4.0f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy, args.sw, args.sh, 2.0f);
    pixel += post_sample_buffer_border(args.src, sx - step_x, sy + step_y, args.sw, args.sh);
    pixel += post_sample_buffer_border(args.src, sx, sy + step_y, args.sw, args.sh, 2.0f);
    pixel += post_sample_buffer_border(args.src, sx + step_x, sy + step_y, args.sw, args.sh);

    pixel *= 1.0f / 20.0f;
    pixel *= args.sa;

    float base_pixel = __ldcs(args.base + x + y * args.tw);
    base_pixel *= args.sb;

    pixel += base_pixel;

    __stcs(args.dst + x + y * args.tw, pixel);
  }
}

#endif /* CU_POST_COMMON_H */
