#ifndef LUMINARY_TEXTURE_H
#define LUMINARY_TEXTURE_H

#include "utils.h"

enum TextureDataType { TexDataFP32 = 0, TexDataUINT8 = 1, TexDataUINT16 = 2 } typedef TextureDataType;
enum TextureWrappingMode { TexModeWrap = 0, TexModeClamp = 1, TexModeMirror = 2, TexModeBorder = 3 } typedef TextureWrappingMode;
enum TextureDimensionType { Tex2D = 0, Tex3D = 1 } typedef TextureDimensionType;
enum TextureStorageLocation { TexStorageCPU = 0, TexStorageGPU = 1 } typedef TextureStorageLocation;
enum TextureFilterMode { TexFilterPoint = 0, TexFilterLinear = 1 } typedef TextureFilterMode;
enum TextureMipmapMode { TexMipmapNone = 0, TexMipmapGenerate = 1 } typedef TextureMipmapMode;
enum TextureReadMode { TexReadModeNormalized = 0, TexReadModeElement = 1 } typedef TextureReadMode;

struct Texture {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t pitch;
  TextureDataType type;
  TextureWrappingMode wrap_mode_S;
  TextureWrappingMode wrap_mode_T;
  TextureWrappingMode wrap_mode_R;
  TextureDimensionType dim;
  TextureStorageLocation storage;
  TextureFilterMode filter;
  TextureMipmapMode mipmap;
  TextureReadMode read_mode;
  uint32_t mipmap_max_level;
  void* data;
  float gamma;
  uint32_t num_components;
} typedef Texture;

LuminaryResult texture_create(
  Texture** tex, uint32_t width, uint32_t height, uint32_t depth, uint32_t pitch, void* data, TextureDataType type, uint32_t num_components,
  TextureStorageLocation storage);
LuminaryResult texture_destroy(Texture** tex);

#endif /* LUMINARY_TEXTURE_H */
