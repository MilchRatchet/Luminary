#ifndef PNG_H
#define PNG_H

#include <stdint.h>

#include "texture.h"
#include "utils.h"

enum PNGColortype {
  PNG_COLORTYPE_GRAYSCALE       = 0,
  PNG_COLORTYPE_TRUECOLOR       = 2,
  PNG_COLORTYPE_INDEXED         = 3,
  PNG_COLORTYPE_GRAYSCALE_ALPHA = 4,
  PNG_COLORTYPE_TRUECOLOR_ALPHA = 6
} typedef PNGColortype;

enum PNGBitdepth {
  PNG_BITDEPTH_1  = 1,
  PNG_BITDEPTH_2  = 2,
  PNG_BITDEPTH_4  = 4,
  PNG_BITDEPTH_8  = 8,
  PNG_BITDEPTH_16 = 16
} typedef PNGBitdepth;

// There is only one method defined each. The idea was to allow for other methods in future standards but that never happened.
enum PNGCompressionMethod { PNG_COMPRESSION_METHOD_0 = 0 } typedef PNGCompressionMethod;
enum PNGFilterMethod { PNG_FILTER_METHOD_0 = 0 } typedef PNGFilterMethod;

enum PNGInterlaceMethod { PNG_INTERLACE_OFF = 0, PNG_INTERLACE_ADAM7 = 1 } typedef PNGInterlaceMethod;

enum PNGFilterFunction {
  PNG_FILTER_NONE    = 0,
  PNG_FILTER_SUB     = 1,
  PNG_FILTER_UP      = 2,
  PNG_FILTER_AVERAGE = 3,
  PNG_FILTER_PAETH   = 4
} typedef PNGFilterFunction;

LuminaryResult png_store_XRGB8(const char* filename, const XRGB8* image, const int width, const int height);
LuminaryResult png_store(
  const char* filename, const uint8_t* image, const uint32_t image_length, const uint32_t width, const uint32_t height,
  const PNGColortype color_type, const PNGBitdepth bit_depth);
LuminaryResult png_load(const uint8_t* file, const size_t file_length, const char* hint_name, Texture** texture);
LuminaryResult png_load_from_file(const char* filename, Texture** texture);

#endif /* PNG_H */
