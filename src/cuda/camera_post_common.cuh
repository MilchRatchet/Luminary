#ifndef CU_CAMERA_POST_COMMON_H
#define CU_CAMERA_POST_COMMON_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "utils.h"

__device__ RGBAhalf sample_pixel(const RGBAhalf* image, float x, float y, const int width, const int height) {
  const float source_x = fmaxf(0.0f, x * (width - 1) - 0.5f);
  const float source_y = fmaxf(0.0f, y * (height - 1) - 0.5f);

  const int index_x = source_x;
  const int index_y = source_y;

  const int index_00 = index_x + index_y * width;
  const int index_01 = index_00 + ((index_y < height - 1) ? width : 0);
  const int index_10 = index_00 + ((index_x < width - 1) ? 1 : 0);
  const int index_11 = index_01 + ((index_x < width - 1) ? 1 : 0);

  const RGBAhalf pixel_00 = load_RGBAhalf(image + index_00);
  const RGBAhalf pixel_01 = load_RGBAhalf(image + index_01);
  const RGBAhalf pixel_10 = load_RGBAhalf(image + index_10);
  const RGBAhalf pixel_11 = load_RGBAhalf(image + index_11);

  const float fx  = source_x - index_x;
  const float ifx = 1.0f - fx;
  const float fy  = source_y - index_y;
  const float ify = 1.0f - fy;

  const float f00 = ifx * ify;
  const float f01 = ifx * fy;
  const float f10 = fx * ify;
  const float f11 = fx * fy;

  RGBAhalf result = scale_RGBAhalf(pixel_00, (__half) f00);
  result          = fma_RGBAhalf(pixel_01, (__half) f01, result);
  result          = fma_RGBAhalf(pixel_10, (__half) f10, result);
  result          = fma_RGBAhalf(pixel_11, (__half) f11, result);

  return result;
}

__device__ RGBAhalf sample_pixel_clamp(const RGBAhalf* image, float x, float y, const int width, const int height) {
  x = fmaxf(x, 0.0f);
  y = fmaxf(y, 0.0f);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

  return sample_pixel(image, x, y, width, height);
}

__device__ RGBAhalf sample_pixel_border(const RGBAhalf* image, float x, float y, const int width, const int height) {
  if (x > __uint_as_float(0b00111111011111111111111111111111) || x < 0.0f) {
    return get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
  }

  if (y > __uint_as_float(0b00111111011111111111111111111111) || y < 0.0f) {
    return get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);
  }

  return sample_pixel(image, x, y, width, height);
}

__global__ void image_downsample(RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf a1 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a2 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a3 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBAhalf a4 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBAhalf pixel = add_RGBAhalf(add_RGBAhalf(a1, a2), add_RGBAhalf(a3, a4));

    pixel = add_RGBAhalf(pixel, sample_pixel_clamp(source, sx, sy, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy - step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy + step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy + step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy + step_y, sw, sh), (__half) 0.25f));

    pixel = scale_RGBAhalf(pixel, (__half) 0.125f);

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

__global__ void image_downsample_threshold(
  RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th, const float thresh) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf a1 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a2 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a3 = sample_pixel_clamp(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBAhalf a4 = sample_pixel_clamp(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBAhalf pixel = add_RGBAhalf(add_RGBAhalf(a1, a2), add_RGBAhalf(a3, a4));

    pixel = add_RGBAhalf(pixel, sample_pixel_clamp(source, sx, sy, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy - step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy + step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy + step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy + step_y, sw, sh), (__half) 0.25f));

    pixel = scale_RGBAhalf(pixel, (__half) 0.125f);

    pixel = max_RGBAhalf(sub_RGBAhalf(pixel, get_RGBAhalf(thresh, thresh, thresh, thresh)), get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f));

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

__global__ void image_upsample(
  RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, RGBAhalf* base, const int tw, const int th, const float sa,
  const float sb) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf pixel = sample_pixel_clamp(source, sx - step_x, sy - step_y, sw, sh);

    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy - step_y, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel_clamp(source, sx + step_x, sy - step_y, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx - step_x, sy, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy, sw, sh), 4.0f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx + step_x, sy, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel_clamp(source, sx - step_x, sy + step_y, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel_clamp(source, sx, sy + step_y, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel_clamp(source, sx + step_x, sy + step_y, sw, sh));

    pixel = scale_RGBAhalf(pixel, (__half) (0.0625f * sa));

    RGBAhalf base_pixel = load_RGBAhalf(base + x + y * tw);
    base_pixel          = scale_RGBAhalf(base_pixel, sb);
    pixel               = add_RGBAhalf(pixel, base_pixel);

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_CAMERA_POST_COMMON_H */
