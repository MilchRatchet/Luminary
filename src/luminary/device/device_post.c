#include "device_post.h"

#include "device.h"
#include "device_constant_memory_manager.h"
#include "device_memory.h"
#include "internal_error.h"
#include "kernel.h"
#include "kernel_args.h"

////////////////////////////////////////////////////////////////////
// Glow
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

static uint32_t _device_post_glow_mip_count(const uint32_t width, const uint32_t height) {
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

static LuminaryResult _device_post_glow_create(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  post->glow_mip_count = _device_post_glow_mip_count(post->width, post->height);

  __FAILURE_HANDLE(host_malloc(&post->glow_mips, sizeof(float*) * post->glow_mip_count));

  uint32_t width  = post->width;
  uint32_t height = post->height;

  for (uint32_t i = 0; i < post->glow_mip_count; i++) {
    width  = width >> 1;
    height = height >> 1;
    __FAILURE_HANDLE(device_malloc((void**) &post->glow_mips[i], sizeof(float) * width * height));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_glow_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  const uint32_t undersampling_stage = (device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  // If we are undersampling too much we can't properly do mip-chain based glow.
  if (undersampling_stage + 1 >= post->glow_mip_count)
    return LUMINARY_SUCCESS;

  const uint32_t width  = post->width >> undersampling_stage;
  const uint32_t height = post->height >> undersampling_stage;

  const uint32_t mip_count = post->glow_mip_count - undersampling_stage;

  for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
    float strength = 0.0f;
    if (post->glow_mode == LUMINARY_GLOW_MODE_MONOCHROMATIC) {
      strength = post->glow_strength;
    }
    else if (post->glow_mode == LUMINARY_GLOW_MODE_TRICHROMATIC) {
      if (channel_id == FRAME_CHANNEL_RED)
        strength = post->glow_color.r;
      else if (channel_id == FRAME_CHANNEL_GREEN)
        strength = post->glow_color.g;
      else if (channel_id == FRAME_CHANNEL_BLUE)
        strength = post->glow_color.b;

      strength *= post->glow_strength;
    }

    if (strength <= 0.0f)
      continue;

    {
      KernelArgsPostImageDownsample args;

      args.src       = DEVICE_PTR(device->work_buffers->frame_result[channel_id]);
      args.sw        = width;
      args.sh        = height;
      args.dst       = DEVICE_PTR(post->glow_mips[0]);
      args.tw        = width >> 1;
      args.th        = height >> 1;
      args.threshold = 0.0f;

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_POST_IMAGE_DOWNSAMPLE], &args, device->stream_main));
    }

    for (uint32_t i = 0; i < mip_count - 1; i++) {
      KernelArgsPostImageDownsample args;

      args.src       = DEVICE_PTR(post->glow_mips[i]);
      args.sw        = width >> (i + 1);
      args.sh        = height >> (i + 1);
      args.dst       = DEVICE_PTR(post->glow_mips[i + 1]);
      args.tw        = width >> (i + 2);
      args.th        = height >> (i + 2);
      args.threshold = 0.0f;

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_POST_IMAGE_DOWNSAMPLE], &args, device->stream_main));
    }

    for (uint32_t i = mip_count - 1; i > 0; i--) {
      KernelArgsPostImageUpsample args;

      args.src  = DEVICE_PTR(post->glow_mips[i]);
      args.sw   = width >> (i + 1);
      args.sh   = height >> (i + 1);
      args.dst  = DEVICE_PTR(post->glow_mips[i - 1]);
      args.base = DEVICE_PTR(post->glow_mips[i - 1]);
      args.tw   = width >> i;
      args.th   = height >> i;
      args.sa   = 1.0f;
      args.sb   = 1.0f;

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
    }

    {
      KernelArgsPostImageUpsample args;

      args.src  = DEVICE_PTR(post->glow_mips[0]);
      args.sw   = width >> 1;
      args.sh   = height >> 1;
      args.dst  = DEVICE_PTR(device->work_buffers->frame_result[channel_id]);
      args.base = DEVICE_PTR(device->work_buffers->frame_result[channel_id]);
      args.tw   = width;
      args.th   = height;
      args.sa   = strength / mip_count;
      args.sb   = 1.0f - strength;

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_POST_IMAGE_UPSAMPLE], &args, device->stream_main));
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_post_glow_destroy(DevicePost* post) {
  __CHECK_NULL_ARGUMENT(post);

  if (post->glow_mips == NULL)
    return LUMINARY_SUCCESS;

  for (uint32_t i = 0; i < post->glow_mip_count; i++) {
    __FAILURE_HANDLE(device_free(&post->glow_mips[i]));
  }

  __FAILURE_HANDLE(host_free(&post->glow_mips));
  post->glow_mip_count = 0;

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

    if (post->glow_active) {
      __FAILURE_HANDLE(_device_post_glow_destroy(post));
      __FAILURE_HANDLE(_device_post_glow_create(post));
    }
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_update(DevicePost* post, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(camera);

  bool camera_glow = false;
  if (camera->glow_mode == LUMINARY_GLOW_MODE_MONOCHROMATIC) {
    camera_glow = camera->glow_strength > 0.0f;
  }
  else if (camera->glow_mode == LUMINARY_GLOW_MODE_TRICHROMATIC) {
    camera_glow =
      (camera->glow_strength > 0.0f) && ((camera->glow_color.r > 0.0f) || (camera->glow_color.g > 0.0f) || (camera->glow_color.b > 0.0f));
  }

  post->glow_mode     = camera->glow_mode;
  post->glow_strength = camera->glow_strength;
  post->glow_color    = camera->glow_color;

  if (post->glow_active != camera_glow) {
    post->glow_active = camera_glow;

    if (post->glow_active) {
      __FAILURE_HANDLE(_device_post_glow_create(post));
    }

    if (post->glow_active == false) {
      __FAILURE_HANDLE(_device_post_glow_destroy(post));
    }
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_apply(DevicePost* post, Device* device) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(device);

  const DeviceConstantMemory* constant_memory;
  __FAILURE_HANDLE(device_constant_memory_manager_get_host_buffer(device->constant_memory, &constant_memory));

  if (constant_memory->settings.shading_mode != LUMINARY_SHADING_MODE_DEFAULT)
    return LUMINARY_SUCCESS;

  if (constant_memory->settings.adaptive_sampling_output_mode != LUMINARY_ADAPTIVE_SAMPLING_OUTPUT_MODE_BEAUTY)
    return LUMINARY_SUCCESS;

  if (constant_memory->settings.window_width != constant_memory->settings.width)
    return LUMINARY_SUCCESS;

  if (constant_memory->settings.window_height != constant_memory->settings.height)
    return LUMINARY_SUCCESS;

  if (post->glow_active) {
    __FAILURE_HANDLE(_device_post_glow_apply(post, device));
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_post_destroy(DevicePost** post) {
  __CHECK_NULL_ARGUMENT(post);
  __CHECK_NULL_ARGUMENT(*post);

  __FAILURE_HANDLE(_device_post_glow_destroy(*post));

  __FAILURE_HANDLE(host_free(post));

  return LUMINARY_SUCCESS;
}
