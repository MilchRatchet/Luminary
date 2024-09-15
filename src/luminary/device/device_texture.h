#ifndef LUMINARY_DEVICE_TEXTURE_H
#define LUMINARY_DEVICE_TEXTURE_H

#include "device_utils.h"
#include "texture.h"

struct DeviceTexture {
  cudaTextureObject_t tex;
  float inv_width;
  float inv_height;
  float gamma;
} typedef DeviceTexture;

LuminaryResult device_texture_create(DeviceTexture** device_texture, Texture* texture);
LuminaryResult device_texture_destroy(DeviceTexture** device_texture);

#endif /* LUMINARY_DEVICE_TEXTURE_H */
