#ifndef CU_CAMERA_POST_COMMON_H
#define CU_CAMERA_POST_COMMON_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "utils.h"

LUM_DEVICE_FUNC RGBF sample_pixel(const RGBF* image, float x, float y, const int width, const int height) {
  const float source_x = fmaxf(0.0f, x * (width - 1) - 0.5f);
  const float source_y = fmaxf(0.0f, y * (height - 1) - 0.5f);

  const int index_x = source_x;
  const int index_y = source_y;

  const int index_00 = index_x + index_y * width;
  const int index_01 = index_00 + ((index_y < height - 1) ? width : 0);
  const int index_10 = index_00 + ((index_x < width - 1) ? 1 : 0);
  const int index_11 = index_01 + ((index_x < width - 1) ? 1 : 0);

  const RGBF pixel_00 = load_RGBF(image + index_00);
  const RGBF pixel_01 = load_RGBF(image + index_01);
  const RGBF pixel_10 = load_RGBF(image + index_10);
  const RGBF pixel_11 = load_RGBF(image + index_11);

  const float fx  = source_x - index_x;
  const float ifx = 1.0f - fx;
  const float fy  = source_y - index_y;
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

LUM_DEVICE_FUNC RGBF sample_pixel_clamp(const RGBF* image, float x, float y, const int width, const int height) {
  x = fmaxf(x, 0.0f);
  y = fmaxf(y, 0.0f);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

  return sample_pixel(image, x, y, width, height);
}

LUM_DEVICE_FUNC RGBF sample_pixel_border(const RGBF* image, float x, float y, const int width, const int height) {
  if (x > __uint_as_float(0b00111111011111111111111111111111) || x < 0.0f) {
    return get_color(0.0f, 0.0f, 0.0f);
  }

  if (y > __uint_as_float(0b00111111011111111111111111111111) || y < 0.0f) {
    return get_color(0.0f, 0.0f, 0.0f);
  }

  return sample_pixel(image, x, y, width, height);
}

#endif /* CU_CAMERA_POST_COMMON_H */
