#ifndef CU_CAMERA_POST_BLOOM_H
#define CU_CAMERA_POST_BLOOM_H

#include "buffer.h"
#include "camera_post_common.cuh"
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
 *
 * I also based this implementation on a nice blog post by Lena Piquet:
 * https://www.froyok.fr/blog/2021-12-ue4-custom-bloom/
 */

static int _bloom_mip_count(const int width, const int height) {
  int min_dim = min(width, height);

  if (min_dim == 0)
    return 0;

  int i = 0;

  while (min_dim != 1) {
    i++;
    min_dim = min_dim >> 1;
  }

  return i;
}

extern "C" void device_bloom_init(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  const int mip_count = _bloom_mip_count(width, height);

  instance->bloom_mips_gpu   = (RGBF**) malloc(sizeof(RGBF*) * mip_count);
  instance->bloom_mips_count = mip_count;

  for (int i = 0; i < mip_count; i++) {
    width  = width >> 1;
    height = height >> 1;
    device_malloc((void**) &(instance->bloom_mips_gpu[i]), sizeof(RGBF) * width * height);
  }
}

extern "C" void device_bloom_apply(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  const int width  = instance->output_width;
  const int height = instance->output_height;

  int mip_count = instance->bloom_mips_count;

  if (mip_count == 0) {
    device_bloom_init(instance);

    mip_count = instance->bloom_mips_count;

    if (mip_count == 0) {
      crash_message("Failed to initialized bloom post process.");
    }
  }

  const float blend = instance->scene.camera.bloom_blend;

  image_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width, height, instance->bloom_mips_gpu[0], width >> 1, height >> 1);

  for (int i = 0; i < mip_count - 1; i++) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> (i + 2);
    const int th = height >> (i + 2);
    image_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(instance->bloom_mips_gpu[i], sw, sh, instance->bloom_mips_gpu[i + 1], tw, th);
  }

  for (int i = mip_count - 1; i > 0; i--) {
    const int sw = width >> (i + 1);
    const int sh = height >> (i + 1);
    const int tw = width >> i;
    const int th = height >> i;
    image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      instance->bloom_mips_gpu[i], sw, sh, instance->bloom_mips_gpu[i - 1], instance->bloom_mips_gpu[i - 1], tw, th, 1.0f, 1.0f);
  }

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->bloom_mips_gpu[0], width >> 1, height >> 1, dst, dst, width, height, blend / mip_count, 1.0f - blend);
}

extern "C" void device_bloom_clear(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  const int mip_count = _bloom_mip_count(width, height);

  for (int i = 0; i < mip_count; i++) {
    width  = width >> 1;
    height = height >> 1;
    device_free(instance->bloom_mips_gpu[i], sizeof(RGBF) * width * height);
  }

  free(instance->bloom_mips_gpu);
  instance->bloom_mips_count = 0;
}

#endif /* CU_CAMERA_POST_BLOOM_H */
