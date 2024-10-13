#ifndef LUMINARY_DEVICE_TEXTURE_H
#define LUMINARY_DEVICE_TEXTURE_H

#include "device_memory.h"
#include "device_utils.h"
#include "texture.h"

struct DeviceTexture {
  DEVICE void* memory;
  CUtexObject tex;
  float inv_width;
  float inv_height;
  float gamma;
  bool is_3D;
};

LuminaryResult device_texture_create(DeviceTexture** device_texture, Texture* texture, CUstream stream);
LuminaryResult device_texture_destroy(DeviceTexture** device_texture);

#endif /* LUMINARY_DEVICE_TEXTURE_H */
