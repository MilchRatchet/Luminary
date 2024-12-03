#ifndef LUMINARY_DEVICE_BSDF_H
#define LUMINARY_DEVICE_BSDF_H

#include "device_texture.h"
#include "device_utils.h"

struct Device typedef Device;

struct BSDFLUT {
  Texture* conductor;
  Texture* specular;
  Texture* dielectric;
  Texture* dielectric_inv;
} typedef BSDFLUT;

struct DeviceBSDFLUT {
  DeviceTexture* conductor;
  DeviceTexture* specular;
  DeviceTexture* dielectric;
  DeviceTexture* dielectric_inv;
} typedef DeviceBSDFLUT;

LuminaryResult bsdf_lut_create(BSDFLUT** lut);
DEVICE_CTX_FUNC LuminaryResult bsdf_lut_generate(BSDFLUT* lut, Device* device);
LuminaryResult bsdf_lut_destroy(BSDFLUT** lut);

DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_create(DeviceBSDFLUT** lut);
DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_update(DeviceBSDFLUT* lut, Device* device, const BSDFLUT* source_lut);
DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_destroy(DeviceBSDFLUT** lut);

#endif /* LUMINARY_DEVICE_BSDF_H */
