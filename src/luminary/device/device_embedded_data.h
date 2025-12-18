#ifndef LUMINARY_DEVICE_EMBEDDED_DATA_H
#define LUMINARY_DEVICE_EMBEDDED_DATA_H

#include "device_utils.h"

struct Device typedef Device;

struct DeviceEmbeddedDataPtrs {
  CUdeviceptr bluenoise_1D;
  CUdeviceptr bluenoise_2D;
  CUdeviceptr bridge_lut;
  CUdeviceptr spectral_cdf;
  DeviceTextureObject moon_albedo_tex;
  DeviceTextureObject moon_normal_tex;
  DeviceTextureObject spectral_xy_tex;
  DeviceTextureObject spectral_z_tex;
} typedef DeviceEmbeddedDataPtrs;

struct DeviceEmbeddedData {
  DEVICE uint16_t* bluenoise_1D;
  DEVICE uint32_t* bluenoise_2D;
  DEVICE float* bridge_lut;
  DEVICE float* spectral_cdf;
  DeviceTexture* moon_albedo_tex;
  DeviceTexture* moon_normal_tex;
  DeviceTexture* spectral_xy_tex;
  DeviceTexture* spectral_z_tex;
} typedef DeviceEmbeddedData;

LuminaryResult device_embedded_data_create(DeviceEmbeddedData** data);
DEVICE_CTX_FUNC LuminaryResult device_embedded_data_update(DeviceEmbeddedData* data, Device* device, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_embedded_data_get_ptrs(DeviceEmbeddedData* data, DeviceEmbeddedDataPtrs* ptrs);
LuminaryResult device_embedded_data_destroy(DeviceEmbeddedData** data);

#endif /* LUMINARY_DEVICE_EMBEDDED_DATA_H */
