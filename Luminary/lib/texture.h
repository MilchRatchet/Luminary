#ifndef TEXTURE_H
#define TEXTURE_H

#include "image.h"

struct TextureAssignment {
  uint16_t albedo_map;
  uint16_t illuminance_map;
  uint16_t material_map;
  uint16_t _p;
} typedef TextureAssignment;

struct TextureG {
  unsigned int width;
  unsigned int height;
  float* data;
} typedef TextureG;

enum TextureDataType { TexDataFP32 = 0, TexDataUINT8 = 1 } typedef TextureDataType;

struct TextureRGBA {
  unsigned int width;
  unsigned int height;
  TextureDataType type;
  void* data;
} typedef TextureRGBA;

#endif /* TEXTURE_H */
