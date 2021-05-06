#ifndef TEXTURE_H
#define TEXTURE_H

#include "image.h"

struct texture_assignment {
  uint16_t albedo_map;
  uint16_t illuminance_map;
  uint16_t material_map;
  uint16_t _p;
} typedef texture_assignment;

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
