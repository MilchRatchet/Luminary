#ifndef LUMINARY_DEVICE_CLOUD_H
#define LUMINARY_DEVICE_CLOUD_H

#include "device_texture.h"
#include "device_utils.h"

struct Device typedef Device;

struct DeviceCloudNoise {
  DeviceTexture* shape_tex;
  DeviceTexture* detail_tex;
  DeviceTexture* weather_tex;
  bool initialized;
  uint32_t seed;
} typedef DeviceCloudNoise;

DEVICE_CTX_FUNC LuminaryResult device_cloud_noise_create(DeviceCloudNoise** cloud_noise, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_cloud_noise_generate(DeviceCloudNoise* cloud_noise, const Cloud* cloud, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_cloud_noise_destroy(DeviceCloudNoise** cloud_noise);

#endif /* LUMINARY_DEVICE_CLOUD_H */
