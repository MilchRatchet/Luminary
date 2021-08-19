#ifndef CU_BLOOM_H
#define CU_BLOOM_H

#include "math.cuh"
#include "utils.cuh"

#define BLOOM_MIP_COUNT 9

__device__ RGBF sample_pixel(RGBF* image, float x, float y, const int width, const int height) {
  x = __saturatef(x);
  y = __saturatef(y);

  // clamp with largest float below 1.0f
  x = fminf(x, __uint_as_float(0b00111111011111111111111111111111));
  y = fminf(y, __uint_as_float(0b00111111011111111111111111111111));

  const float source_x = x * (width - 1);
  const float source_y = y * (height - 1);

  const int index_x = source_x;
  const int index_y = source_y;

  const int index_0 = index_x + index_y * width;
  const int index_1 = index_x + (index_y + 1) * width;

  const RGBF pixel_00 = image[index_0];
  const RGBF pixel_01 = image[index_0 + 1];
  const RGBF pixel_10 = image[index_1];
  const RGBF pixel_11 = image[index_1 + 1];

  const float fx  = source_x - index_x;
  const float ifx = 1.0f - fx;
  const float fy  = source_y - index_y;
  const float ify = 1.0f - fy;

  const float f00 = ifx * ify;
  const float f10 = fx * ify;
  const float f01 = ifx * fy;
  const float f11 = fx * fy;

  return get_color(
    pixel_00.r * f00 + pixel_01.r * f01 + pixel_10.r * f10 + pixel_11.r * f11,
    pixel_00.g * f00 + pixel_01.g * f01 + pixel_10.g * f10 + pixel_11.g * f11,
    pixel_00.b * f00 + pixel_01.b * f01 + pixel_10.b * f10 + pixel_11.b * f11);
}

__global__ void bloom_downsample(
  RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th, const float threshold) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 0.5f / (sw - 1);
  const float step_y  = 0.5f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF a1 = sample_pixel(source, sx - 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a2 = sample_pixel(source, sx + 0.5f * step_x, sy - 0.5f * step_y, sw, sh);
    RGBF a3 = sample_pixel(source, sx - 0.5f * step_x, sy + 0.5f * step_y, sw, sh);
    RGBF a4 = sample_pixel(source, sx + 0.5f * step_x, sy + 0.5f * step_y, sw, sh);

    RGBF pixel = add_color(add_color(a1, a2), add_color(a3, a4));

    pixel = add_color(pixel, sample_pixel(source, sx, sy, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx, sy - step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx - step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx + step_x, sy, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx, sy + step_y, sw, sh), 0.5f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx - step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx + step_x, sy - step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx - step_x, sy + step_y, sw, sh), 0.25f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx + step_x, sy + step_y, sw, sh), 0.25f));

    pixel = scale_color(pixel, 0.125f);

    float lum = luminance(pixel);
    pixel     = scale_color(pixel, 1.0f / (lum + eps));
    lum       = (fmaxf(0.0f, lum - threshold));
    pixel     = scale_color(pixel, lum);

    target[x + y * tw] = pixel;

    id += blockDim.x * gridDim.x;
  }
}

__global__ void bloom_upsample(RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);
  const float step_x  = 0.5f / (sw - 1);
  const float step_y  = 0.5f / (sh - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x;
    const float sy = scale_y * y;

    RGBF pixel = sample_pixel(source, sx - step_x, sy - step_y, sw, sh);

    pixel = add_color(pixel, scale_color(sample_pixel(source, sx, sy - step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel(source, sx + step_x, sy - step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx - step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx, sy, sw, sh), 4.0f));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx + step_x, sy, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel(source, sx - step_x, sy + step_y, sw, sh));
    pixel = add_color(pixel, scale_color(sample_pixel(source, sx, sy + step_y, sw, sh), 2.0f));
    pixel = add_color(pixel, sample_pixel(source, sx + step_x, sy + step_y, sw, sh));

    pixel = scale_color(pixel, 0.0625f);

    RGBF o = target[x + y * tw];
    pixel  = add_color(pixel, o);

    target[x + y * tw] = pixel;

    id += blockDim.x * gridDim.x;
  }
}

extern "C" void apply_bloom(RaytraceInstance* instance, RGBF* image) {
  const int width  = instance->width;
  const int height = instance->height;

  if (width < (1 << BLOOM_MIP_COUNT) || height < (1 << BLOOM_MIP_COUNT))
    return;

  bloom_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    image, width, height, instance->bloom_mips_gpu[0], width >> 1, height >> 1, instance->scene_gpu.camera.bloom_threshold);

  for (int i = 0; i < BLOOM_MIP_COUNT - 1; i++) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> (i + 2);
    const int th = height >> (i + 2);
    bloom_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      instance->bloom_mips_gpu[i], sw, sh, instance->bloom_mips_gpu[i + 1], tw, th, 0.0f);
  }

  for (int i = BLOOM_MIP_COUNT - 1; i > 0; i--) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> i;
    const int th = height >> i;
    bloom_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(instance->bloom_mips_gpu[i], sw, sh, instance->bloom_mips_gpu[i - 1], tw, th);
  }

  bloom_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(instance->bloom_mips_gpu[0], width >> 1, height >> 1, image, width, height);
}

static void allocate_bloom_mips(RaytraceInstance* instance) {
  int width  = instance->width;
  int height = instance->height;

  instance->bloom_mips_gpu = (RGBF**) malloc(sizeof(RGBF*) * BLOOM_MIP_COUNT);

  for (int i = 0; i < BLOOM_MIP_COUNT; i++) {
    width  = width >> 1;
    height = height >> 1;
    gpuErrchk(cudaMalloc((void**) &(instance->bloom_mips_gpu[i]), sizeof(RGBF) * width * height));
  }
}

static void free_bloom_mips(RaytraceInstance* instance) {
  for (int i = 0; i < BLOOM_MIP_COUNT; i++) {
    gpuErrchk(cudaFree(instance->bloom_mips_gpu[i]));
  }

  free(instance->bloom_mips_gpu);
}

#endif /* CU_BLOOM_H */
