#include "device_sky.h"

#include "device.h"
#include "device_texture.h"
#include "internal_error.h"
#include "sky.h"

LuminaryResult sky_hdri_create(SkyHDRI** hdri) {
  __CHECK_NULL_ARGUMENT(hdri);

  __FAILURE_HANDLE(host_malloc(hdri, sizeof(SkyHDRI)));

  (*hdri)->sky_is_dirty    = true;
  (*hdri)->output_is_dirty = true;
  (*hdri)->id              = 0;

  __FAILURE_HANDLE(sample_count_set(&(*hdri)->sample_count, 16));
  __FAILURE_HANDLE(sky_get_default(&(*hdri)->sky));

  __FAILURE_HANDLE(texture_create(&(*hdri)->color_tex, 128, 128, 1, 128 * sizeof(RGBAF), (void*) 0, TexDataFP32, 4));
  __FAILURE_HANDLE(texture_create(&(*hdri)->shadow_tex, 128, 128, 1, 128 * sizeof(float), (void*) 0, TexDataFP32, 1));

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

      __FAILURE_HANDLE(texture_create(&hdri->color_tex, width, height, 1, width * sizeof(RGBAF), (void*) 0, TexDataFP32, 4));
      __FAILURE_HANDLE(texture_create(&hdri->shadow_tex, width, height, 1, width * sizeof(float), (void*) 0, TexDataFP32, 1));

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

      __FAILURE_HANDLE(host_malloc(&hdri->color_tex->data, hdri->color_tex->pitch * hdri->color_tex->height));
      __FAILURE_HANDLE(host_malloc(&hdri->shadow_tex->data, hdri->shadow_tex->pitch * hdri->shadow_tex->height));
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

DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_update(DeviceSkyHDRI* hdri, Device* device, const SkyHDRI* source_hdri, bool* has_changed) {
  __CHECK_NULL_ARGUMENT(hdri);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(source_hdri);

  *has_changed = false;

  if (source_hdri->id != hdri->reference_id) {
    if (hdri->color_tex) {
      __FAILURE_HANDLE(device_texture_destroy(&hdri->color_tex));
    }

    if (hdri->shadow_tex) {
      __FAILURE_HANDLE(device_texture_destroy(&hdri->shadow_tex));
    }

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

  if ((*hdri)->color_tex) {
    __FAILURE_HANDLE(device_texture_destroy(&(*hdri)->color_tex));
  }

  if ((*hdri)->shadow_tex) {
    __FAILURE_HANDLE(device_texture_destroy(&(*hdri)->shadow_tex));
  }

  __FAILURE_HANDLE(host_free(hdri));

  return LUMINARY_SUCCESS;
}
