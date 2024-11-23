#include "device_post.h"

#include "device_memory.h"
#include "internal_error.h"

////////////////////////////////////////////////////////////////////
// Bloom
////////////////////////////////////////////////////////////////////

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

static uint32_t _device_post_bloom_mip_count(const uint32_t width, const uint32_t height) {
  uint32_t min_dim = min(width, height);

  if (min_dim == 0)
    return 0;

  uint32_t i = 0;

  while (min_dim != 1) {
    i++;
    min_dim = min_dim >> 1;
  }

  return i;
}

static LuminaryResult _device_post_bloom_create(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  post->bloom_mip_count = _device_post_bloom_mip_count(post->width, post->height);

  __FAILURE_HANDLE(host_malloc(&post->bloom_mips, sizeof(RGBF*) * post->bloom_mip_count));

  uint32_t width  = post->width;
  uint32_t height = post->height;

  for (uint32_t i = 0; i < post->bloom_mip_count; i++) {
    width  = width >> 1;
    height = height >> 1;
    __FAILURE_HANDLE(device_malloc((void**) &post->bloom_mips[i], sizeof(RGBF) * width * height));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_bloom_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  /*
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
  */

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_bloom_destroy(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  if (!post->bloom_mips)
    return LUMINARY_SUCCESS;

  for (uint32_t i = 0; i < post->bloom_mip_count; i++) {
    __FAILURE_HANDLE(device_free(post->bloom_mips[i]));
  }

  __FAILURE_HANDLE(host_free(&post->bloom_mips));
  post->bloom_mip_count = 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Lens Flare
////////////////////////////////////////////////////////////////////

#define LENS_FLARE_NUM_BUFFERS 4

static LuminaryResult _device_post_lens_flare_create(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  __FAILURE_HANDLE(host_malloc(&post->lens_flare_buffers, sizeof(RGBF*) * LENS_FLARE_NUM_BUFFERS));

  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[0], sizeof(RGBF) * post->width * post->height));
  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[1], sizeof(RGBF) * (post->width >> 1) * (post->height >> 1)));
  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[2], sizeof(RGBF) * (post->width >> 1) * (post->height >> 1)));
  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[3], sizeof(RGBF) * (post->width >> 2) * (post->height >> 2)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_lens_flare_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  /*
static void _lens_flare_apply_ghosts(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  int width  = instance->internal_width;
  int height = instance->internal_height;

  _lens_flare_ghosts<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
}

static void _lens_flare_apply_halo(RaytraceInstance* instance, const RGBF* src, RGBF* dst) {
  int width  = instance->internal_width;
  int height = instance->internal_height;

  _lens_flare_halo<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
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
  */

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_lens_flare_destroy(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  if (!post->lens_flare_buffers)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[0]));
  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[1]));
  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[2]));
  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[3]));

  __FAILURE_HANDLE(host_free(&post->lens_flare_buffers));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////

DEVICE_CTX_FUNC LuminaryResult device_post_create(DevicePost** post) {
  __CHECK_NULL_ARGUMENT(post);

  __FAILURE_HANDLE(host_malloc(post, sizeof(DevicePost)));
  memset(*post, 0, sizeof(DevicePost));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_allocate(DevicePost* post, uint32_t width, uint32_t height) {
  __CHECK_NULL_ARGUMENT(post);

  if ((post->width != width) || (post->height != height)) {
    post->width  = width;
    post->height = height;

    if (post->bloom) {
      __FAILURE_HANDLE(_device_post_bloom_destroy(post));
      __FAILURE_HANDLE(_device_post_bloom_create(post));
    }

    if (post->lens_flare) {
      __FAILURE_HANDLE(_device_post_lens_flare_destroy(post));
      __FAILURE_HANDLE(_device_post_lens_flare_create(post));
    }
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_update(DevicePost* post, Camera* camera) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(camera);

  post->bloom_blend          = camera->bloom_blend;
  post->lens_flare_threshold = camera->lens_flare_threshold;

  if (post->bloom != camera->bloom) {
    post->bloom = camera->bloom;

    if (post->bloom) {
      __FAILURE_HANDLE(_device_post_bloom_create(post));
    }

    if (!post->bloom) {
      __FAILURE_HANDLE(_device_post_bloom_destroy(post));
    }
  }

  if (post->lens_flare != camera->lens_flare) {
    post->lens_flare = camera->lens_flare;

    if (post->lens_flare) {
      __FAILURE_HANDLE(_device_post_lens_flare_create(post));
    }

    if (!post->lens_flare) {
      __FAILURE_HANDLE(_device_post_lens_flare_destroy(post));
    }
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  if (post->bloom) {
    __FAILURE_HANDLE(_device_post_bloom_apply(post, device));
  }

  if (post->lens_flare) {
    __FAILURE_HANDLE(_device_post_lens_flare_apply(post, device));
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_destroy(DevicePost** post) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(*post);

  __FAILURE_HANDLE(_device_post_bloom_destroy(*post));
  __FAILURE_HANDLE(_device_post_lens_flare_destroy(*post));

  __FAILURE_HANDLE(host_free(post));

  return LUMINARY_SUCCESS;
}
