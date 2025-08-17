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

__device__ RGBAF sample_pixel_border(const RGBF* image, float x, float y, const int width, const int height) {
  if (x > __uint_as_float(0b00111111011111111111111111111111) || x < 0.0f) {
    return RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
  }

  if (y > __uint_as_float(0b00111111011111111111111111111111) || y < 0.0f) {
    return RGBAF_set(0.0f, 0.0f, 0.0f, 0.0f);
  }

  return transparent_color(sample_pixel(image, x, y, width, height), 1.0f);
}

LUMINARY_KERNEL void camera_post_image_downsample(const KernelArgsCameraPostImageDownsample args) {
  uint32_t id = THREAD_ID;

  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);
  const float step_x  = 1.0f / (args.sw - 1);
  const float step_y  = 1.0f / (args.sh - 1);

  while (id < args.tw * args.th) {
    const uint32_t y = id / args.tw;
    const uint32_t x = id - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    const RGBAF a1 = sample_pixel_border(args.src, sx - 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    const RGBAF a2 = sample_pixel_border(args.src, sx + 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    const RGBAF a3 = sample_pixel_border(args.src, sx - 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);
    const RGBAF a4 = sample_pixel_border(args.src, sx + 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);

    RGBAF pixel = RGBAF_add(RGBAF_add(a1, a2), RGBAF_add(a3, a4));

    pixel = RGBAF_add(pixel, sample_pixel_border(args.src, sx, sy, args.sw, args.sh));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy - step_y, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy + step_y, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy - step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy - step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy + step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy + step_y, args.sw, args.sh), 0.25f));

    const RGBF result = (pixel.a > 0.0f) ? scale_color(opaque_color(pixel), 1.0f / pixel.a) : splat_color(0.0f);

    store_RGBF(args.dst, x + y * args.tw, result);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void camera_post_image_downsample_threshold(const KernelArgsCameraPostImageDownsampleThreshold args) {
  uint32_t id = THREAD_ID;

  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);
  const float step_x  = 1.0f / (args.sw - 1);
  const float step_y  = 1.0f / (args.sh - 1);

  while (id < args.tw * args.th) {
    const uint32_t y = id / args.tw;
    const uint32_t x = id - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    const RGBAF a1 = sample_pixel_border(args.src, sx - 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    const RGBAF a2 = sample_pixel_border(args.src, sx + 0.5f * step_x, sy - 0.5f * step_y, args.sw, args.sh);
    const RGBAF a3 = sample_pixel_border(args.src, sx - 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);
    const RGBAF a4 = sample_pixel_border(args.src, sx + 0.5f * step_x, sy + 0.5f * step_y, args.sw, args.sh);

    RGBAF pixel = RGBAF_add(RGBAF_add(a1, a2), RGBAF_add(a3, a4));

    pixel = RGBAF_add(pixel, sample_pixel_border(args.src, sx, sy, args.sw, args.sh));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy - step_y, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy + step_y, args.sw, args.sh), 0.5f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy - step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy - step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy + step_y, args.sw, args.sh), 0.25f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy + step_y, args.sw, args.sh), 0.25f));

    RGBF result = (pixel.a > 0.0f) ? scale_color(opaque_color(pixel), 1.0f / pixel.a) : splat_color(0.0f);
    result      = max_color(sub_color(result, splat_color(args.threshold)), splat_color(0.0f));

    store_RGBF(args.dst, x + y * args.tw, result);

    id += blockDim.x * gridDim.x;
  }
}

LUMINARY_KERNEL void camera_post_image_upsample(const KernelArgsCameraPostImageUpsample args) {
  uint32_t id = THREAD_ID;

  const float scale_x = 1.0f / (args.tw - 1);
  const float scale_y = 1.0f / (args.th - 1);
  const float step_x  = 1.0f / (args.sw - 1);
  const float step_y  = 1.0f / (args.sh - 1);

  while (id < args.tw * args.th) {
    const uint32_t y = id / args.tw;
    const uint32_t x = id - y * args.tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBAF pixel = sample_pixel_border(args.src, sx - step_x, sy - step_y, args.sw, args.sh);

    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy - step_y, args.sw, args.sh), 2.0f));
    pixel = RGBAF_add(pixel, sample_pixel_border(args.src, sx + step_x, sy - step_y, args.sw, args.sh));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx - step_x, sy, args.sw, args.sh), 2.0f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy, args.sw, args.sh), 4.0f));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx + step_x, sy, args.sw, args.sh), 2.0f));
    pixel = RGBAF_add(pixel, sample_pixel_border(args.src, sx - step_x, sy + step_y, args.sw, args.sh));
    pixel = RGBAF_add(pixel, RGBAF_scale(sample_pixel_border(args.src, sx, sy + step_y, args.sw, args.sh), 2.0f));
    pixel = RGBAF_add(pixel, sample_pixel_border(args.src, sx + step_x, sy + step_y, args.sw, args.sh));

    RGBF result = (pixel.a > 0.0f) ? scale_color(opaque_color(pixel), args.sa / pixel.a) : splat_color(0.0f);

    RGBF base_pixel = load_RGBF(args.base + x + y * args.tw);
    base_pixel      = scale_color(base_pixel, args.sb);
    result          = add_color(result, base_pixel);

    store_RGBF(args.dst, x + y * args.tw, result);

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_CAMERA_POST_COMMON_H */
