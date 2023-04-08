#ifndef TEXTURE_H
#define TEXTURE_H

#include "buffer.h"
#include "structs.h"

#if __cplusplus
extern "C" {
#endif

struct cudaExtent texture_make_cudaextent(size_t depth, size_t height, size_t width);
struct cudaPitchedPtr texture_make_cudapitchedptr(void* ptr, size_t pitch, size_t xsize, size_t ysize);
enum cudaTextureAddressMode texture_get_address_mode(TextureWrappingMode tex);
enum cudaTextureReadMode texture_get_read_mode(TextureRGBA* tex);
enum cudaTextureFilterMode texture_get_filter_mode(TextureRGBA* tex);
enum cudaMemcpyKind texture_get_copy_to_device_type(TextureRGBA* tex);
struct cudaChannelFormatDesc texture_get_channel_format_desc(TextureRGBA* tex);
size_t texture_get_pixel_size(TextureRGBA* tex);
void texture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length);
void texture_free_atlas(DeviceBuffer* texture_atlas, const int textures_length);
void texture_create(
  TextureRGBA* tex, unsigned int width, unsigned int height, unsigned int depth, unsigned int pitch, void* data, TextureDataType type,
  TextureStorageLocation storage);

#if __cplusplus
}
#endif

#endif /* TEXTURE_H */
