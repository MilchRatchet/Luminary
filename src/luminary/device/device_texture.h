#ifndef LUMINARY_DEVICE_TEXTURE_H
#define LUMINARY_DEVICE_TEXTURE_H

#include "device_memory.h"
#include "device_utils.h"
#include "texture.h"

struct DeviceTexture {
  void* memory;
  CUtexObject tex;
  uint16_t width;
  uint16_t height;
  uint16_t depth;
  float gamma;
  bool is_3D;
  size_t pitch;
  size_t pixel_size;
} typedef DeviceTexture;

DEVICE_CTX_FUNC LuminaryResult device_texture_create(DeviceTexture** device_texture, const Texture* texture, CUstream stream);
DEVICE_CTX_FUNC LuminaryResult device_texture_copy_from_mem(DeviceTexture* device_texture, const DEVICE void* mem, CUstream stream);
DEVICE_CTX_FUNC LuminaryResult device_texture_destroy(DeviceTexture** device_texture);

#endif /* LUMINARY_DEVICE_TEXTURE_H */
