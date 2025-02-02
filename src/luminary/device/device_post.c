#include "device_post.h"

#include "device.h"
#include "device_memory.h"
#include "internal_error.h"
#include "kernel.h"
#include "kernel_args.h"

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

  const uint32_t width  = post->width;
  const uint32_t height = post->height;

  {
    KernelArgsCameraPostImageDownsample args;

    args.src = DEVICE_PTR(device->buffers.frame_accumulate);
    args.sw  = width;
    args.sh  = height;
    args.dst = DEVICE_PTR(post->bloom_mips[0]);
    args.tw  = width >> 1;
    args.th  = height >> 1;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE], &args, device->stream_main));
  }

  for (uint32_t i = 0; i < post->bloom_mip_count - 1; i++) {
    KernelArgsCameraPostImageDownsample args;

    args.src = DEVICE_PTR(post->bloom_mips[i]);
    args.sw  = width >> (i + 1);
    args.sh  = height >> (i + 1);
    args.dst = DEVICE_PTR(post->bloom_mips[i + 1]);
    args.tw  = width >> (i + 2);
    args.th  = height >> (i + 2);

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE], &args, device->stream_main));
  }

  for (uint32_t i = post->bloom_mip_count - 1; i > 0; i--) {
    KernelArgsCameraPostImageUpsample args;

    args.src  = DEVICE_PTR(post->bloom_mips[i]);
    args.sw   = width >> (i + 1);
    args.sh   = height >> (i + 1);
    args.dst  = DEVICE_PTR(post->bloom_mips[i - 1]);
    args.base = DEVICE_PTR(post->bloom_mips[i - 1]);
    args.tw   = width >> i;
    args.th   = height >> i;
    args.sa   = 1.0f;
    args.sb   = 1.0f;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostImageUpsample args;

    args.src  = DEVICE_PTR(post->bloom_mips[0]);
    args.sw   = width >> 1;
    args.sh   = height >> 1;
    args.dst  = DEVICE_PTR(device->buffers.frame_accumulate);
    args.base = DEVICE_PTR(device->buffers.frame_accumulate);
    args.tw   = width;
    args.th   = height;
    args.sa   = post->bloom_blend / post->bloom_mip_count;
    args.sb   = 1.0f - post->bloom_blend;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_bloom_destroy(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  if (!post->bloom_mips)
    return LUMINARY_SUCCESS;

  for (uint32_t i = 0; i < post->bloom_mip_count; i++) {
    __FAILURE_HANDLE(device_free(&post->bloom_mips[i]));
  }

  __FAILURE_HANDLE(host_free(&post->bloom_mips));
  post->bloom_mip_count = 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Lens Flare
////////////////////////////////////////////////////////////////////

#define LENS_FLARE_NUM_BUFFERS 2

static LuminaryResult _device_post_lens_flare_create(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  __FAILURE_HANDLE(host_malloc(&post->lens_flare_buffers, sizeof(RGBF*) * LENS_FLARE_NUM_BUFFERS));

  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[0], sizeof(RGBF) * (post->width >> 1) * (post->height >> 1)));
  __FAILURE_HANDLE(device_malloc((void**) &post->lens_flare_buffers[1], sizeof(RGBF) * (post->width >> 1) * (post->height >> 1)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_lens_flare_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  const uint32_t width  = post->width;
  const uint32_t height = post->height;

  {
    KernelArgsCameraPostImageDownsampleThreshold args;

    args.src       = DEVICE_PTR(device->buffers.frame_accumulate);
    args.sw        = width;
    args.sh        = height;
    args.dst       = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.tw        = width >> 1;
    args.th        = height >> 1;
    args.threshold = post->lens_flare_threshold;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE_THRESHOLD], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostImageDownsample args;

    args.src = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.sw  = width >> 1;
    args.sh  = height >> 1;
    args.dst = DEVICE_PTR(post->lens_flare_buffers[1]);
    args.tw  = width >> 2;
    args.th  = height >> 2;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_DOWNSAMPLE], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostImageUpsample args;

    args.src  = DEVICE_PTR(post->lens_flare_buffers[1]);
    args.sw   = width >> 2;
    args.sh   = height >> 2;
    args.base = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.dst  = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.tw   = width >> 1;
    args.th   = height >> 1;
    args.sa   = 0.5f;
    args.sb   = 0.5f;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostLensFlareGhosts args;

    args.src = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.sw  = width >> 1;
    args.sh  = height >> 1;
    args.dst = DEVICE_PTR(post->lens_flare_buffers[1]);
    args.tw  = width >> 1;
    args.th  = height >> 1;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_GHOSTS], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostLensFlareHalo args;

    args.src = DEVICE_PTR(post->lens_flare_buffers[1]);
    args.sw  = width >> 1;
    args.sh  = height >> 1;
    args.dst = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.tw  = width >> 1;
    args.th  = height >> 1;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_LENS_FLARE_HALO], &args, device->stream_main));
  }

  {
    KernelArgsCameraPostImageUpsample args;

    args.src  = DEVICE_PTR(post->lens_flare_buffers[0]);
    args.sw   = width >> 1;
    args.sh   = height >> 1;
    args.base = DEVICE_PTR(device->buffers.frame_accumulate);
    args.dst  = DEVICE_PTR(device->buffers.frame_accumulate);
    args.tw   = width;
    args.th   = height;
    args.sa   = 1.0f;
    args.sb   = 1.0f;

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_CAMERA_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_lens_flare_destroy(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  if (!post->lens_flare_buffers)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[0]));
  __FAILURE_HANDLE(device_free((void**) &post->lens_flare_buffers[1]));

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

DEVICE_CTX_FUNC LuminaryResult device_post_update(DevicePost* post, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(camera);

  const bool camera_bloom = camera->bloom_blend > 0.0f;

  post->bloom_blend          = camera->bloom_blend;
  post->lens_flare_threshold = camera->lens_flare_threshold;

  if (post->bloom != camera_bloom) {
    post->bloom = camera_bloom;

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

  if (device->constant_memory->settings.shading_mode != LUMINARY_SHADING_MODE_DEFAULT) {
    return LUMINARY_SUCCESS;
  }

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
