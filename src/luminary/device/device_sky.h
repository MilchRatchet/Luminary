#ifndef LUMINARY_DEVICE_SKY_H
#define LUMINARY_DEVICE_SKY_H

#include "device_utils.h"
#include "texture.h"

struct SkyHDRI {
  bool sky_is_dirty;
  bool output_is_dirty;
  Sky sky;
  uint32_t id;
  SampleCountSlice sample_count;
  Texture* color_tex;
  Texture* shadow_tex;
} typedef SkyHDRI;

struct DeviceSkyHDRI {
  uint32_t reference_id;
  DeviceTexture* color_tex;
  DeviceTexture* shadow_tex;
} typedef DeviceSkyHDRI;

struct Device typedef Device;

LuminaryResult sky_hdri_create(SkyHDRI** hdri);
LuminaryResult sky_hdri_update(SkyHDRI* hdri, const Sky* sky);
DEVICE_CTX_FUNC LuminaryResult sky_hdri_generate(SkyHDRI* hdri, Device* device);
LuminaryResult sky_hdri_destroy(SkyHDRI** hdri);

DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_create(DeviceSkyHDRI** hdri);
DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_update(DeviceSkyHDRI* hdri, Device* device, const SkyHDRI* source_hdri, bool* has_changed);
DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_destroy(DeviceSkyHDRI** hdri);

#endif /* LUMINARY_DEVICE_SKY_H */
