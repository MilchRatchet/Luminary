#ifndef TEXTURE_H
#define TEXTURE_H

#include "buffer.h"
#include "structs.h"

#if __cplusplus
extern "C" {
#endif

void texture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length);
void texture_free_atlas(DeviceBuffer* texture_atlas, const int textures_length);
void texture_create(
  TextureRGBA* tex, unsigned int width, unsigned int height, unsigned int depth, unsigned int pitch, void* data, TextureDataType type,
  TextureStorageLocation storage);

#if __cplusplus
}
#endif

#endif /* TEXTURE_H */
