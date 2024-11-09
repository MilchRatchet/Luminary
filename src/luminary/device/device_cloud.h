#ifndef LUMINARY_DEVICE_CLOUD_H
#define LUMINARY_DEVICE_CLOUD_H

#include "device_texture.h"
#include "device_utils.h"

struct Device typedef Device;

struct CloudNoise {
  DeviceTexture* shape_tex;
  DeviceTexture* detail_tex;
  DeviceTexture* weather_tex;
} typedef CloudNoise;

DEVICE_CTX_FUNC LuminaryResult device_cloud_noise_create(CloudNoise** cloud_noise, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_cloud_noise_destroy(CloudNoise** cloud_noise, Device* device);

#endif /* LUMINARY_DEVICE_CLOUD_H */
