#ifndef TEXTURE_H
#define TEXTURE_H

#include "buffer.h"
#include "structs.h"

#if __cplusplus
extern "C" {
#endif

void texture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length);
void texture_free_atlas(DeviceBuffer* texture_atlas, const int textures_length);

#if __cplusplus
}
#endif

#endif /* TEXTURE_H */
