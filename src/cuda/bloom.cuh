#ifndef CU_BLOOM_H
#define CU_BLOOM_H

#include "math.cuh"
#include "memory.cuh"
#include "utils.cuh"

/*
 * This implementation is based on a presentation
 * "Next Generation Post Processing in Call of Duty: Advanced Warfare"
 * by Jorge Jimenez at SIGGRAPH 2014.
 *
 * We downsample using a 13 tap box filter.
 * We upsample using a 9 tap tent filter.
 *
 * For performance it may be beneficial to wrap the mip buffers into a texture object for better sampling
 */

#define BLOOM_MIP_COUNT 9

__device__ RGBAhalf sample_pixel(const RGBAhalf* image, float x, float y, const int width, const int height) {
  x = fmaxf(x, 0.0f);
  y = fmaxf(y, 0.0f);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

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

__global__ void bloom_downsample(RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
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

    RGBAhalf a1 = sample_pixel(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a2 = sample_pixel(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a3 = sample_pixel(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBAhalf a4 = sample_pixel(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBAhalf pixel = add_RGBAhalf(add_RGBAhalf(a1, a2), add_RGBAhalf(a3, a4));

    pixel = add_RGBAhalf(pixel, sample_pixel(source, sx, sy, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy - step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy + step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy + step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy + step_y, sw, sh), (__half) 0.25f));

    pixel = scale_RGBAhalf(pixel, (__half) 0.125f);

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

__global__ void bloom_downsample_truncate(RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float thresh = device_scene.camera.bloom_threshold;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 1.0f / (sw - 1);
  const float step_y  = 1.0f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf a1 = sample_pixel(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a2 = sample_pixel(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBAhalf a3 = sample_pixel(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBAhalf a4 = sample_pixel(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBAhalf pixel = add_RGBAhalf(add_RGBAhalf(a1, a2), add_RGBAhalf(a3, a4));

    pixel = add_RGBAhalf(pixel, sample_pixel(source, sx, sy, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy - step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy + step_y, sw, sh), (__half) 0.5f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy - step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy + step_y, sw, sh), (__half) 0.25f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy + step_y, sw, sh), (__half) 0.25f));

    pixel = scale_RGBAhalf(pixel, (__half) 0.125f);

    store_RGBAhalf(
      target + x + y * tw, bound_RGBAhalf(max_RGBAhalf(
                             sub_RGBAhalf(pixel, get_RGBAhalf(thresh, thresh, thresh, thresh)), get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f))));

    id += blockDim.x * gridDim.x;
  }
}

__global__ void bloom_upsample(
  RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, RGBAhalf* base, const int tw, const int th, const float a,
  const float b) {
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

    RGBAhalf pixel = sample_pixel(source, sx - step_x, sy - step_y, sw, sh);

    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy - step_y, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel(source, sx + step_x, sy - step_y, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx - step_x, sy, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy, sw, sh), 4.0f));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx + step_x, sy, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel(source, sx - step_x, sy + step_y, sw, sh));
    pixel = add_RGBAhalf(pixel, scale_RGBAhalf(sample_pixel(source, sx, sy + step_y, sw, sh), (__half) 2.0f));
    pixel = add_RGBAhalf(pixel, sample_pixel(source, sx + step_x, sy + step_y, sw, sh));

    pixel = scale_RGBAhalf(pixel, (__half) 0.0625f);

    RGBAhalf o = load_RGBAhalf(base + x + y * tw);
    o          = scale_RGBAhalf(o, a);
    pixel      = scale_RGBAhalf(pixel, b);
    pixel      = add_RGBAhalf(pixel, o);

    store_RGBAhalf(base + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

extern "C" void apply_bloom(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst) {
  const int width  = instance->width;
  const int height = instance->height;

  int mip_count_w;
  int mip_count_h;
  bsr(width, mip_count_w);
  bsr(height, mip_count_h);

  const int mip_count = min(BLOOM_MIP_COUNT, min(mip_count_w, mip_count_h));

  bloom_downsample_truncate<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (RGBAhalf*) src, width, height, (RGBAhalf*) instance->bloom_mips_gpu[0], width >> 1, height >> 1);

  for (int i = 0; i < mip_count - 1; i++) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> (i + 2);
    const int th = height >> (i + 2);
    bloom_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      (RGBAhalf*) instance->bloom_mips_gpu[i], sw, sh, (RGBAhalf*) instance->bloom_mips_gpu[i + 1], tw, th);
  }

  for (int i = mip_count - 1; i > 0; i--) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> i;
    const int th = height >> i;
    bloom_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      (RGBAhalf*) instance->bloom_mips_gpu[i], sw, sh, (RGBAhalf*) instance->bloom_mips_gpu[i - 1],
      (RGBAhalf*) instance->bloom_mips_gpu[i - 1], tw, th, 1.0f, 1.0f);
  }

  bloom_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    (RGBAhalf*) instance->bloom_mips_gpu[0], width >> 1, height >> 1, (RGBAhalf*) dst, (RGBAhalf*) src, width, height, 1.0f,
    0.01f * instance->scene_gpu.camera.bloom_strength / mip_count);
}

static void allocate_bloom_mips(RaytraceInstance* instance) {
  int width  = instance->width;
  int height = instance->height;

  instance->bloom_mips_gpu = (RGBAhalf**) malloc(sizeof(RGBAhalf*) * BLOOM_MIP_COUNT);

  for (int i = 0; i < BLOOM_MIP_COUNT; i++) {
    width  = width >> 1;
    height = height >> 1;
    device_malloc((void**) &(instance->bloom_mips_gpu[i]), sizeof(RGBAhalf) * width * height);
  }
}

static void free_bloom_mips(RaytraceInstance* instance) {
  int width  = instance->width;
  int height = instance->height;

  for (int i = 0; i < BLOOM_MIP_COUNT; i++) {
    width  = width >> 1;
    height = height >> 1;
    device_free(instance->bloom_mips_gpu[i], sizeof(RGBAhalf) * width * height);
  }

  free(instance->bloom_mips_gpu);
}

#endif /* CU_BLOOM_H */
