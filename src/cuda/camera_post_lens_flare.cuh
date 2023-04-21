#ifndef CU_CAMERA_POST_LENS_FLARE_H
#define CU_CAMERA_POST_LENS_FLARE_H

#include "utils.h"

#define LENS_FLARE_NUM_BUFFERS 3

/*
 * I based this implementation on a nice blog post by Lena Piquet:
 * https://www.froyok.fr/blog/2021-09-ue4-custom-lens-flare/
 */

extern "C" void device_lens_flare_init(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  instance->lens_flare_buffers_gpu = (RGBAhalf**) malloc(sizeof(RGBAhalf*) * LENS_FLARE_NUM_BUFFERS);

  device_malloc((void**) &(instance->lens_flare_buffers_gpu[0]), sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[1]), sizeof(RGBAhalf) * (width >> 2) * (height >> 2));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[2]), sizeof(RGBAhalf) * width * height);
}

__global__ void _lens_flare_ghosts(RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int ghost_count           = 8;
  const float ghost_scales[]      = {-1.0f, -0.5f, -0.25f, -2.0f, -3.0f, -4.0f, 2.0f, 0.25f};
  const float ghost_intensities[] = {1.0f, 0.4f, 0.2f, 0.7f, 0.5f, 0.3f, 0.5f, 0.2f};

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf pixel = load_RGBAhalf(target + x + y * tw);

    for (int i = 0; i < ghost_count; i++) {
      const float sxi = ((sx - 0.5f) * ghost_scales[i]) + 0.5f;
      const float syi = ((sy - 0.5f) * ghost_scales[i]) + 0.5f;

      RGBAhalf ghost = sample_pixel_border(source, sxi, syi, sw, sh);

      ghost = scale_RGBAhalf(ghost, ghost_intensities[i] * 0.0005f);

      pixel = add_RGBAhalf(pixel, ghost);
    }

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

static void _lens_flare_apply_ghosts(RaytraceInstance* instance, RGBAhalf* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  _lens_flare_ghosts<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(instance->lens_flare_buffers_gpu[2], width, height, dst, width, height);
}

extern "C" void device_lens_flare_apply(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  if (!instance->lens_flare_buffers_gpu) {
    device_lens_flare_init(instance);

    if (!instance->lens_flare_buffers_gpu) {
      crash_message("Failed to initialize lens flare buffers.");
    }
  }

  image_downsample_threshold<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    src, width, height, instance->lens_flare_buffers_gpu[0], width >> 1, height >> 1, instance->scene.camera.lens_flare_threshold);

  image_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[0], width >> 1, height >> 1, instance->lens_flare_buffers_gpu[1], width >> 2, height >> 2);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[1], width >> 2, height >> 2, instance->lens_flare_buffers_gpu[0], instance->lens_flare_buffers_gpu[0],
    width >> 1, height >> 1, 0.5f, 0.5f);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[0], width >> 1, height >> 1, instance->lens_flare_buffers_gpu[2], instance->lens_flare_buffers_gpu[2],
    width, height, 1.0f, 0.0f);

  _lens_flare_apply_ghosts(instance, dst);
}

extern "C" void device_lens_flare_clear(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  device_free(instance->lens_flare_buffers_gpu[0], sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_free(instance->lens_flare_buffers_gpu[1], sizeof(RGBAhalf) * (width >> 2) * (height >> 2));
  device_free(instance->lens_flare_buffers_gpu[2], sizeof(RGBAhalf) * width * height);

  free(instance->lens_flare_buffers_gpu);
}

#endif /* CU_CAMERA_POST_LENS_FLARE_H */
