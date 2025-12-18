#ifndef LUMINARY_DEVICE_TEXTURE_MANAGER_H
#define LUMINARY_DEVICE_TEXTURE_MANAGER_H

#include "device_texture.h"
#include "device_utils.h"

struct Device typedef Device;

struct DeviceTextureManagerPtrs {
  CUdeviceptr textures;
} typedef DeviceTextureManagerPtrs;

struct DeviceTextureManager {
  ARRAY DeviceTexture** textures;
  DEVICE DeviceTextureObject* texture_objs;
} typedef DeviceTextureManager;

LuminaryResult device_texture_manager_create(DeviceTextureManager** manager);
DEVICE_CTX_FUNC LuminaryResult device_texture_manager_add(
  DeviceTextureManager* manager, Device* device, const Texture** textures, uint32_t num_textures, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_texture_manager_get_ptrs(DeviceTextureManager* manager, DeviceTextureManagerPtrs* ptrs);
DEVICE_CTX_FUNC LuminaryResult device_texture_manager_destroy(DeviceTextureManager** manager);

#endif /* LUMINARY_DEVICE_TEXTURE_MANAGER_H */
