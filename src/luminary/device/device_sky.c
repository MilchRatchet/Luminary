#include "device_sky.h"

#include "device.h"
#include "device_texture.h"
#include "internal_error.h"
#include "kernel_args.h"
#include "sky.h"
#include "sky_defines.h"

LuminaryResult sky_lut_create(SkyLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_malloc(lut, sizeof(SkyLUT)));

  (*lut)->sky_is_dirty = true;
  (*lut)->id           = 0;

  __FAILURE_HANDLE(sky_get_default(&(*lut)->sky));

  void* transmittance_low_data;
  __FAILURE_HANDLE(host_malloc(&transmittance_low_data, SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT * sizeof(RGBAF)));

  void* transmittance_high_data;
  __FAILURE_HANDLE(host_malloc(&transmittance_high_data, SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT * sizeof(RGBAF)));

  void* multiscattering_low_data;
  __FAILURE_HANDLE(host_malloc(&multiscattering_low_data, SKY_MS_TEX_SIZE * SKY_MS_TEX_SIZE * sizeof(RGBAF)));

  void* multiscattering_high_data;
  __FAILURE_HANDLE(host_malloc(&multiscattering_high_data, SKY_MS_TEX_SIZE * SKY_MS_TEX_SIZE * sizeof(RGBAF)));

  __FAILURE_HANDLE(
    texture_create(&(*lut)->transmittance_low, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, transmittance_low_data, TexDataFP32, 4));
  __FAILURE_HANDLE(
    texture_create(&(*lut)->transmittance_high, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT, 1, transmittance_high_data, TexDataFP32, 4));
  __FAILURE_HANDLE(
    texture_create(&(*lut)->multiscattering_low, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, multiscattering_low_data, TexDataFP32, 4));
  __FAILURE_HANDLE(
    texture_create(&(*lut)->multiscattering_high, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1, multiscattering_high_data, TexDataFP32, 4));

  (*lut)->transmittance_low->wrap_mode_R = TexModeClamp;
  (*lut)->transmittance_low->wrap_mode_S = TexModeClamp;
  (*lut)->transmittance_low->wrap_mode_T = TexModeClamp;

  (*lut)->transmittance_high->wrap_mode_R = TexModeClamp;
  (*lut)->transmittance_high->wrap_mode_S = TexModeClamp;
  (*lut)->transmittance_high->wrap_mode_T = TexModeClamp;

  (*lut)->multiscattering_low->wrap_mode_R = TexModeClamp;
  (*lut)->multiscattering_low->wrap_mode_S = TexModeClamp;
  (*lut)->multiscattering_low->wrap_mode_T = TexModeClamp;

  (*lut)->multiscattering_high->wrap_mode_R = TexModeClamp;
  (*lut)->multiscattering_high->wrap_mode_S = TexModeClamp;
  (*lut)->multiscattering_high->wrap_mode_T = TexModeClamp;

  return LUMINARY_SUCCESS;
}

LuminaryResult sky_lut_update(SkyLUT* lut, const Sky* sky) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(sky);

  bool is_dirty = false;
  __FAILURE_HANDLE(sky_check_for_dirty(sky, &lut->sky, &is_dirty));

  if (is_dirty) {
    memcpy(&lut->sky, sky, sizeof(Sky));
    lut->sky_is_dirty = true;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _sky_lut_generate_lut(SkyLUT* lut, DeviceSkyLUT* device_lut, Device* device) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);

  KernelArgsSkyComputeTransmittanceLUT transmission_lut_args;
  transmission_lut_args.dst_low  = DEVICE_PTR(device_lut->transmittance_low->memory);
  transmission_lut_args.dst_high = DEVICE_PTR(device_lut->transmittance_high->memory);

  KernelArgsSkyComputeMultiscatteringLUT multiscattering_lut_args;
  multiscattering_lut_args.dst_low  = DEVICE_PTR(device_lut->multiscattering_low->memory);
  multiscattering_lut_args.dst_high = DEVICE_PTR(device_lut->multiscattering_high->memory);

  __FAILURE_HANDLE(device_struct_texture_object_convert(device_lut->transmittance_low, &multiscattering_lut_args.transmission_low_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(device_lut->transmittance_high, &multiscattering_lut_args.transmission_high_tex));

  __FAILURE_HANDLE(device_sync_constant_memory(device));
  __FAILURE_HANDLE(kernel_execute_with_args(
    device->cuda_kernels[CUDA_KERNEL_TYPE_SKY_COMPUTE_TRANSMITTANCE_LUT], &transmission_lut_args, device->stream_main));
  __FAILURE_HANDLE(kernel_execute_custom(
    device->cuda_kernels[CUDA_KERNEL_TYPE_SKY_COMPUTE_MULTISCATTERING_LUT], SKY_MS_ITER, 1, 1, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE, 1,
    &multiscattering_lut_args, device->stream_main));

  __FAILURE_HANDLE(device_download2D(
    lut->transmittance_low->data, device_lut->transmittance_low->memory, device_lut->transmittance_low->pitch,
    SKY_TM_TEX_WIDTH * sizeof(RGBAF), SKY_TM_TEX_HEIGHT, device->stream_main));
  __FAILURE_HANDLE(device_download2D(
    lut->transmittance_high->data, device_lut->transmittance_high->memory, device_lut->transmittance_high->pitch,
    SKY_TM_TEX_WIDTH * sizeof(RGBAF), SKY_TM_TEX_HEIGHT, device->stream_main));
  __FAILURE_HANDLE(device_download2D(
    lut->multiscattering_low->data, device_lut->multiscattering_low->memory, device_lut->multiscattering_low->pitch,
    SKY_MS_TEX_SIZE * sizeof(RGBAF), SKY_MS_TEX_SIZE, device->stream_main));
  __FAILURE_HANDLE(device_download2D(
    lut->multiscattering_high->data, device_lut->multiscattering_high->memory, device_lut->multiscattering_high->pitch,
    SKY_MS_TEX_SIZE * sizeof(RGBAF), SKY_MS_TEX_SIZE, device->stream_main));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult sky_lut_generate(SkyLUT* lut, Device* device) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);

  if (lut->sky_is_dirty) {
    DeviceSkyLUT* device_lut = device->sky_lut;

    lut->sky_is_dirty = false;
    lut->id++;

    bool has_changed;
    __FAILURE_HANDLE(device_sky_lut_update(device_lut, device, lut, &has_changed));

    __FAILURE_HANDLE(_sky_lut_generate_lut(lut, device_lut, device));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult sky_lut_destroy(SkyLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(*lut);

  __FAILURE_HANDLE(texture_destroy(&(*lut)->transmittance_low));
  __FAILURE_HANDLE(texture_destroy(&(*lut)->transmittance_high));
  __FAILURE_HANDLE(texture_destroy(&(*lut)->multiscattering_low));
  __FAILURE_HANDLE(texture_destroy(&(*lut)->multiscattering_high));

  __FAILURE_HANDLE(host_free(lut));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_sky_lut_create(DeviceSkyLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_malloc(lut, sizeof(DeviceSkyLUT)));

  (*lut)->reference_id = 0;

  (*lut)->transmittance_low    = (DeviceTexture*) 0;
  (*lut)->transmittance_high   = (DeviceTexture*) 0;
  (*lut)->multiscattering_low  = (DeviceTexture*) 0;
  (*lut)->multiscattering_high = (DeviceTexture*) 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_sky_lut_free(DeviceSkyLUT* lut) {
  __CHECK_NULL_ARGUMENT(lut);

  if (lut->transmittance_low) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->transmittance_low));
  }

  if (lut->transmittance_high) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->transmittance_high));
  }

  if (lut->multiscattering_low) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->multiscattering_low));
  }

  if (lut->multiscattering_high) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->multiscattering_high));
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_sky_lut_update(DeviceSkyLUT* lut, Device* device, const SkyLUT* source_lut, bool* has_changed) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(source_lut);

  *has_changed = false;

  if (source_lut->id != lut->reference_id) {
    __FAILURE_HANDLE(_device_sky_lut_free(lut));

    __FAILURE_HANDLE(device_texture_create(&lut->transmittance_low, source_lut->transmittance_low, device->stream_main));
    __FAILURE_HANDLE(device_texture_create(&lut->transmittance_high, source_lut->transmittance_high, device->stream_main));
    __FAILURE_HANDLE(device_texture_create(&lut->multiscattering_low, source_lut->multiscattering_low, device->stream_main));
    __FAILURE_HANDLE(device_texture_create(&lut->multiscattering_high, source_lut->multiscattering_high, device->stream_main));

    lut->reference_id = source_lut->id;

    *has_changed = true;
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_sky_lut_destroy(DeviceSkyLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(*lut);

  __FAILURE_HANDLE(_device_sky_lut_free(*lut));

  __FAILURE_HANDLE(host_free(lut));

  return LUMINARY_SUCCESS;
}

LuminaryResult sky_hdri_create(SkyHDRI** hdri) {
  __CHECK_NULL_ARGUMENT(hdri);

  __FAILURE_HANDLE(host_malloc(hdri, sizeof(SkyHDRI)));

  (*hdri)->sky_is_dirty    = true;
  (*hdri)->output_is_dirty = true;
  (*hdri)->id              = 0;

  __FAILURE_HANDLE(sample_count_set(&(*hdri)->sample_count, 16));
  __FAILURE_HANDLE(sky_get_default(&(*hdri)->sky));

  __FAILURE_HANDLE(texture_create(&(*hdri)->color_tex, 128, 128, 1, (void*) 0, TexDataFP32, 4));
  __FAILURE_HANDLE(texture_create(&(*hdri)->shadow_tex, 128, 128, 1, (void*) 0, TexDataFP32, 1));

  return LUMINARY_SUCCESS;
}

LuminaryResult sky_hdri_update(SkyHDRI* hdri, const Sky* sky) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(sky);

  bool is_dirty = false;
  __FAILURE_HANDLE(sky_check_for_dirty(sky, &hdri->sky, &is_dirty));

  if (is_dirty) {
    memcpy(&hdri->sky, sky, sizeof(Sky));
    hdri->sky_is_dirty = true;

    const uint32_t width  = sky->hdri_dim;
    const uint32_t height = sky->hdri_dim;

    if (hdri->color_tex->width != width || hdri->color_tex->height != height) {
      __FAILURE_HANDLE(texture_destroy(&hdri->color_tex));
      __FAILURE_HANDLE(texture_destroy(&hdri->shadow_tex));

      __FAILURE_HANDLE(texture_create(&hdri->color_tex, width, height, 1, (void*) 0, TexDataFP32, 4));
      __FAILURE_HANDLE(texture_create(&hdri->shadow_tex, width, height, 1, (void*) 0, TexDataFP32, 1));

      hdri->output_is_dirty = true;
    }

    __FAILURE_HANDLE(sample_count_set(&hdri->sample_count, sky->hdri_samples));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _sky_hdri_compute(SkyHDRI* hdri, Device* device) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(device);

  // TODO: Compute the HDRI
  // Currently emulating the side effects
  memset(hdri->color_tex->data, 0, hdri->color_tex->pitch * hdri->color_tex->height);
  memset(hdri->shadow_tex->data, 0, hdri->shadow_tex->pitch * hdri->shadow_tex->height);
  hdri->sample_count.current_sample_count = hdri->sample_count.end_sample_count;

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult sky_hdri_generate(SkyHDRI* hdri, Device* device) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(device);

  if (hdri->sky_is_dirty) {
    hdri->sample_count.current_sample_count = 0;
  }

  const bool requires_rendering = hdri->output_is_dirty || (hdri->sample_count.current_sample_count < hdri->sample_count.end_sample_count);

  if (requires_rendering) {
    if (hdri->output_is_dirty) {
      if (hdri->color_tex->data) {
        __FAILURE_HANDLE(host_free(&hdri->color_tex->data));
      }

      if (hdri->shadow_tex->data) {
        __FAILURE_HANDLE(host_free(&hdri->shadow_tex->data));
      }

      __FAILURE_HANDLE(host_malloc(&hdri->color_tex->data, hdri->color_tex->width * sizeof(RGBAF) * hdri->color_tex->height));
      __FAILURE_HANDLE(host_malloc(&hdri->shadow_tex->data, hdri->shadow_tex->pitch * sizeof(float) * hdri->shadow_tex->height));
    }

    __FAILURE_HANDLE(_sky_hdri_compute(hdri, device));

    hdri->sky_is_dirty = false;

    hdri->id++;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult sky_hdri_destroy(SkyHDRI** hdri) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(*hdri);

  if ((*hdri)->color_tex) {
    __FAILURE_HANDLE(texture_destroy(&(*hdri)->color_tex));
  }

  if ((*hdri)->shadow_tex) {
    __FAILURE_HANDLE(texture_destroy(&(*hdri)->shadow_tex));
  }

  __FAILURE_HANDLE(host_free(hdri));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_sky_hdri_create(DeviceSkyHDRI** hdri) {
  __CHECK_NULL_ARGUMENT(hdri);

  __FAILURE_HANDLE(host_malloc(hdri, sizeof(DeviceSkyHDRI)));

  (*hdri)->reference_id = 0;

  (*hdri)->color_tex  = (DeviceTexture*) 0;
  (*hdri)->shadow_tex = (DeviceTexture*) 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_sky_hdri_free(DeviceSkyHDRI* hdri) {
  __CHECK_NULL_ARGUMENT(hdri);

  if (hdri->color_tex) {
    __FAILURE_HANDLE(device_texture_destroy(&hdri->color_tex));
  }

  if (hdri->shadow_tex) {
    __FAILURE_HANDLE(device_texture_destroy(&hdri->shadow_tex));
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_update(DeviceSkyHDRI* hdri, Device* device, const SkyHDRI* source_hdri, bool* has_changed) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(source_hdri);

  *has_changed = false;

  if (source_hdri->id != hdri->reference_id) {
    __FAILURE_HANDLE(_device_sky_hdri_free(hdri));

    __FAILURE_HANDLE(device_texture_create(&hdri->color_tex, source_hdri->color_tex, device->stream_main));
    __FAILURE_HANDLE(device_texture_create(&hdri->shadow_tex, source_hdri->shadow_tex, device->stream_main));

    hdri->reference_id = source_hdri->id;

    *has_changed = true;
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_destroy(DeviceSkyHDRI** hdri) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(*hdri);

  __FAILURE_HANDLE(_device_sky_hdri_free(*hdri));

  __FAILURE_HANDLE(host_free(hdri));

  return LUMINARY_SUCCESS;
}
