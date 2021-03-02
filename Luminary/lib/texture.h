#ifndef TEXTURE_H
#define TEXTURE_H

#include "image.h"

struct TextureAssignment {
  uint16_t albedo_map;
  uint16_t illuminance_map;
  uint16_t material_map;
} typedef TextureAssignment;

struct TextureG {
  unsigned int width;
  unsigned int height;
  float* data;
} typedef TextureG;

struct TextureRGBA {
  unsigned int width;
  unsigned int height;
  RGBAF* data;
} typedef TextureRGBA;

#endif /* TEXTURE_H */
