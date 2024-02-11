#ifndef CU_CAMERA_POST_H
#define CU_CAMERA_POST_H

#include "camera_post_bloom.cuh"
#include "camera_post_lens_flare.cuh"
#include "utils.h"

#define LENS_FLARE_NUM_BUFFERS 4

extern "C" void device_lens_flare_init(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  instance->lens_flare_buffers_gpu = (RGBF**) malloc(sizeof(RGBF*) * LENS_FLARE_NUM_BUFFERS);

  device_malloc((void**) &(instance->lens_flare_buffers_gpu[0]), sizeof(RGBF) * width * height);
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[1]), sizeof(RGBF) * (width >> 1) * (height >> 1));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[2]), sizeof(RGBF) * (width >> 1) * (height >> 1));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[3]), sizeof(RGBF) * (width >> 2) * (height >> 2));
}

static void _lens_flare_apply_ghosts(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  camera_lens_flare_ghosts<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
}

static void _lens_flare_apply_halo(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  camera_lens_flare_halo<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
}

extern "C" void device_lens_flare_apply(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  if (!instance->lens_flare_buffers_gpu) {
    device_lens_flare_init(instance);

    if (!instance->lens_flare_buffers_gpu) {
      crash_message("Failed to initialize lens flare buffers.");
    }
  }

  image_downsample_threshold<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    src, width, height, instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, instance->scene.camera.lens_flare_threshold);

  image_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, instance->lens_flare_buffers_gpu[3], width >> 2, height >> 2);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[3], width >> 2, height >> 2, instance->lens_flare_buffers_gpu[2], instance->lens_flare_buffers_gpu[2],
    width >> 1, height >> 1, 0.5f, 0.5f);

  _lens_flare_apply_ghosts(instance, instance->lens_flare_buffers_gpu[2], instance->lens_flare_buffers_gpu[1]);
  _lens_flare_apply_halo(instance, instance->lens_flare_buffers_gpu[1], instance->lens_flare_buffers_gpu[2]);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, dst, dst, width, height, 1.0f, 1.0f);
}

extern "C" void device_lens_flare_clear(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  device_free(instance->lens_flare_buffers_gpu[0], sizeof(RGBF) * width * height);
  device_free(instance->lens_flare_buffers_gpu[1], sizeof(RGBF) * (width >> 1) * (height >> 1));
  device_free(instance->lens_flare_buffers_gpu[2], sizeof(RGBF) * (width >> 1) * (height >> 1));
  device_free(instance->lens_flare_buffers_gpu[3], sizeof(RGBF) * (width >> 2) * (height >> 2));

  free(instance->lens_flare_buffers_gpu);
}

extern "C" void device_camera_post_init(RaytraceInstance* instance) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_init(instance);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_init(instance);
  }
}

extern "C" void device_camera_post_apply(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_apply(instance, src, dst);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_apply(instance, src, dst);
  }
}

extern "C" void device_camera_post_clear(RaytraceInstance* instance) {
  if (instance->scene.camera.lens_flare) {
    device_lens_flare_clear(instance);
  }

  if (instance->scene.camera.bloom) {
    device_bloom_clear(instance);
  }
}

#endif /* CU_CAMERA_POST_H */
