#ifndef PNG_H
#define PNG_H

#include <stdint.h>

#include "structs.h"

#define PNG_COLORTYPE_GRAYSCALE 0
#define PNG_COLORTYPE_TRUECOLOR 2
#define PNG_COLORTYPE_INDEXED 3
#define PNG_COLORTYPE_GRAYSCALE_ALPHA 4
#define PNG_COLORTYPE_TRUECOLOR_ALPHA 6

#define PNG_BITDEPTH_1 1
#define PNG_BITDEPTH_2 2
#define PNG_BITDEPTH_4 4
#define PNG_BITDEPTH_8 8
#define PNG_BITDEPTH_16 16

#define PNG_COMPRESSION_METHOD 0

#define PNG_FILTER_METHOD 0

#define PNG_INTERLACE_OFF 0
#define PNG_INTERLACE_ADAM7 1

int store_XRGB8_png(const char* filename, const XRGB8* image, const int width, const int height);
int store_as_png(
  const char* filename, const uint8_t* image, const uint32_t image_length, const uint32_t width, const uint32_t height,
  const uint8_t color_type, const uint8_t bit_depth);
TextureRGBA load_texture_from_png(const char* filename);

#endif /* PNG_H */
