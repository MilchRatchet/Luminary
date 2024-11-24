#ifndef LUMINARY_DEVICE_TEXTURE_H
#define LUMINARY_DEVICE_TEXTURE_H

#include "device_memory.h"
#include "device_utils.h"
#include "texture.h"

struct DeviceTexture {
  DEVICE void* memory;
  CUtexObject tex;
  uint16_t width;
  uint16_t height;
  float gamma;
  bool is_3D;
  size_t pitch;
} typedef DeviceTexture;

LuminaryResult device_texture_create(DeviceTexture** device_texture, const Texture* texture, CUstream stream);
LuminaryResult device_texture_destroy(DeviceTexture** device_texture);

#endif /* LUMINARY_DEVICE_TEXTURE_H */
