#ifndef LUMINARY_DEVICE_SKY_H
#define LUMINARY_DEVICE_SKY_H

#include "device_utils.h"
#include "texture.h"

struct Device typedef Device;

struct SkyLUT {
  bool sky_is_dirty;
  Sky sky;
  uint32_t id;
  Texture* transmittance_low;
  Texture* transmittance_high;
  Texture* multiscattering_low;
  Texture* multiscattering_high;
} typedef SkyLUT;

struct DeviceSkyLUT {
  bool force_device_update;
  uint32_t reference_id;
  DeviceTexture* transmittance_low;
  DeviceTexture* transmittance_high;
  DeviceTexture* multiscattering_low;
  DeviceTexture* multiscattering_high;
} typedef DeviceSkyLUT;

LuminaryResult sky_lut_create(SkyLUT** lut);
LuminaryResult sky_lut_update(SkyLUT* lut, const Sky* sky);
DEVICE_CTX_FUNC LuminaryResult sky_lut_generate(SkyLUT* lut, Device* device);
LuminaryResult sky_lut_destroy(SkyLUT** lut);

DEVICE_CTX_FUNC LuminaryResult device_sky_lut_create(DeviceSkyLUT** lut);
DEVICE_CTX_FUNC LuminaryResult device_sky_lut_update(DeviceSkyLUT* lut, Device* device, const SkyLUT* source_lut, bool* has_changed);
DEVICE_CTX_FUNC LuminaryResult device_sky_lut_destroy(DeviceSkyLUT** lut);

struct SkyHDRI {
  bool sky_is_dirty;
  bool output_is_dirty;
  Sky sky;
  vec3 origin;
  uint32_t id;
  uint32_t width;
  uint32_t height;
  SampleCountSlice sample_count;
  Texture* color_tex;
  Texture* shadow_tex;
} typedef SkyHDRI;

struct DeviceSkyHDRI {
  bool force_device_update;
  uint32_t reference_id;
  DeviceTexture* color_tex;
  DeviceTexture* shadow_tex;
} typedef DeviceSkyHDRI;

LuminaryResult sky_hdri_create(SkyHDRI** hdri);
LuminaryResult sky_hdri_update(SkyHDRI* hdri, const Sky* sky, const Camera* camera);
DEVICE_CTX_FUNC LuminaryResult sky_hdri_generate(SkyHDRI* hdri, Device* device);
LuminaryResult sky_hdri_destroy(SkyHDRI** hdri);

DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_create(DeviceSkyHDRI** hdri);
DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_update(DeviceSkyHDRI* hdri, Device* device, const SkyHDRI* source_hdri, bool* has_changed);
DEVICE_CTX_FUNC LuminaryResult device_sky_hdri_destroy(DeviceSkyHDRI** hdri);

struct SkyStars {
  Star* data;
  uint32_t* offsets;
  uint32_t seed;
  uint32_t count;
} typedef SkyStars;

struct DeviceSkyStars {
  DEVICE Star* data;
  DEVICE uint32_t* offsets;
  uint32_t seed;
  uint32_t count;
} typedef DeviceSkyStars;

LuminaryResult sky_stars_create(SkyStars** stars);
LuminaryResult sky_stars_update(SkyStars* stars, const Sky* sky);
LuminaryResult sky_stars_destroy(SkyStars** stars);

DEVICE_CTX_FUNC LuminaryResult device_sky_stars_create(DeviceSkyStars** stars);
DEVICE_CTX_FUNC LuminaryResult
  device_sky_stars_update(DeviceSkyStars* stars, Device* device, const SkyStars* source_stars, bool* has_changed);
DEVICE_CTX_FUNC LuminaryResult device_sky_stars_destroy(DeviceSkyStars** stars);

#endif /* LUMINARY_DEVICE_SKY_H */
