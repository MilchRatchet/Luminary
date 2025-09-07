#ifndef LUMINARY_TEXTURE_H
#define LUMINARY_TEXTURE_H

#include "utils.h"

enum TextureStatus { TEXTURE_STATUS_NONE, TEXTURE_STATUS_INVALID, TEXTURE_STATUS_ASYNC_LOADING } typedef TextureStatus;
enum TextureDataType { TEXTURE_DATA_TYPE_FP32, TEXTURE_DATA_TYPE_U8, TEXTURE_DATA_TYPE_U16 } typedef TextureDataType;
enum TextureWrappingMode {
  TEXTURE_WRAPPING_MODE_WRAP,
  TEXTURE_WRAPPING_MODE_CLAMP,
  TEXTURE_WRAPPING_MODE_MIRROR,
  TEXTURE_WRAPPING_MODE_BORDER
} typedef TextureWrappingMode;
enum TextureDimensionType { TEXTURE_DIMENSION_TYPE_2D, TEXTURE_DIMENSION_TYPE_3D } typedef TextureDimensionType;
enum TextureFilterMode { TEXTURE_FILTER_MODE_POINT, TEXTURE_FILTER_MODE_LINEAR } typedef TextureFilterMode;
enum TextureMipmapMode { TEXTURE_MIPMAP_MODE_NONE, TEXTURE_MIPMAP_MODE_GENERATE } typedef TextureMipmapMode;
enum TextureReadMode { TEXTURE_READ_MODE_NORMALIZED, TEXTURE_READ_MODE_ELEMENT } typedef TextureReadMode;

struct Texture {
  TextureStatus status;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t pitch;
  TextureDataType type;
  TextureWrappingMode wrap_mode_S;
  TextureWrappingMode wrap_mode_T;
  TextureWrappingMode wrap_mode_R;
  TextureDimensionType dim;
  TextureFilterMode filter;
  TextureMipmapMode mipmap;
  TextureReadMode read_mode;
  void* data;
  float gamma;
  uint32_t num_components;
  void* async_work_data;
} typedef Texture;

LuminaryResult texture_create(Texture** texture);
LuminaryResult texture_fill(
  Texture* tex, uint32_t width, uint32_t height, uint32_t depth, void* data, TextureDataType type, uint32_t num_components);
LuminaryResult texture_invalidate(Texture* texture);
LuminaryResult texture_is_valid(const Texture* texture, bool* is_valid);
LuminaryResult texture_load_async(Texture* texture, Queue* queue, const char* path);
LuminaryResult texture_await(const Texture* texture);
LuminaryResult texture_destroy(Texture** tex);

#endif /* LUMINARY_TEXTURE_H */
