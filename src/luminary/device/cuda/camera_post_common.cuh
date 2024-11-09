#ifndef CU_CAMERA_POST_COMMON_H
#define CU_CAMERA_POST_COMMON_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "utils.h"

__device__ RGBF
  sample_pixel(const RGBF* image, const float x, const float y, const uint32_t width, const uint32_t height, const float mem_scale = 1.0f) {
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

  const RGBF pixel_00 = load_RGBF(image + index_00);
  const RGBF pixel_01 = load_RGBF(image + index_01);
  const RGBF pixel_10 = load_RGBF(image + index_10);
  const RGBF pixel_11 = load_RGBF(image + index_11);

  const float fx  = source_x - index_x0;
  const float ifx = 1.0f - fx;
  const float fy  = source_y - index_y0;
  const float ify = 1.0f - fy;

  const float f00 = ifx * ify;
  const float f01 = ifx * fy;
  const float f10 = fx * ify;
  const float f11 = fx * fy;

  RGBF result = scale_color(pixel_00, f00);
  result      = fma_color(pixel_01, f01, result);
  result      = fma_color(pixel_10, f10, result);
  result      = fma_color(pixel_11, f11, result);

  return result;
}

__device__ RGBF sample_pixel_clamp(const RGBF* image, float x, float y, const int width, const int height, const float mem_scale = 1.0f) {
  x = fmaxf(x, 0.0f);
  y = fmaxf(y, 0.0f);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

  return sample_pixel(image, x, y, width, height, mem_scale);
}

__device__ RGBF sample_pixel_border(const RGBF* image, float x, float y, const int width, const int height) {
  if (x > __uint_as_float(0b00111111011111111111111111111111) || x < 0.0f) {
    return get_color(0.0f, 0.0f, 0.0f);
  }

  if (y > __uint_as_float(0b00111111011111111111111111111111) || y < 0.0f) {
    return get_color(0.0f, 0.0f, 0.0f);
  }

  return sample_pixel(image, x, y, width, height);
}

LUMINARY_KERNEL void camera_post_image_downsample(
  const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th) {
  unsigned int id = THREAD_ID;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int y = id / tw;
    const int x = id - y * tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF a1 = sample_pixel_border(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a2 = sample_pixel_border(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a3 = sample_pixel_border(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBF a4 = sample_pixel_border(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBF pixel = add_color(add_color(a1, a2), add_color(a3, a4));

    pixel = add_color(pixel, sample_pixel_border(source, sx, sy, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy - step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy + step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy + step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy + step_y, sw, sh), 0.25f));

    pixel = scale_color(pixel, 0.125f);

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void camera_post_image_downsample_threshold(
  const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th, const float thresh) {
  unsigned int id = THREAD_ID;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int y = id / tw;
    const int x = id - y * tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF a1 = sample_pixel_border(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a2 = sample_pixel_border(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a3 = sample_pixel_border(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBF a4 = sample_pixel_border(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBF pixel = add_color(add_color(a1, a2), add_color(a3, a4));

    pixel = add_color(pixel, sample_pixel_border(source, sx, sy, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy - step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy + step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy + step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy + step_y, sw, sh), 0.25f));

    pixel = scale_color(pixel, 0.125f);

    pixel = max_color(sub_color(pixel, get_color(thresh, thresh, thresh)), get_color(0.0f, 0.0f, 0.0f));

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void camera_post_image_upsample(
  const RGBF* source, const int sw, const int sh, RGBF* target, const RGBF* base, const int tw, const int th, const float sa,
  const float sb) {
  unsigned int id = THREAD_ID;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int y = id / tw;
    const int x = id - y * tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF pixel = sample_pixel_border(source, sx - step_x, sy - step_y, sw, sh);

    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy - step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_border(source, sx + step_x, sy - step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx - step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy, sw, sh), 4.0f));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx + step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_border(source, sx - step_x, sy + step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel_border(source, sx, sy + step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel_border(source, sx + step_x, sy + step_y, sw, sh));

    pixel = scale_color(pixel, 0.0625f * sa);

    RGBF base_pixel = load_RGBF(base + x + y * tw);
    base_pixel      = scale_color(base_pixel, sb);
    pixel           = add_color(pixel, base_pixel);

    store_RGBF(target + x + y * tw, pixel);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_CAMERA_POST_COMMON_H */
