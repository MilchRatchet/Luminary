#ifndef LUM_QOI_H
#define LUM_QOI_H

#include <stdint.h>

#include "texture.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

#define QOI_COLORTYPE_TRUECOLOR 3
#define QOI_COLORTYPE_TRUECOLOR_ALPHA 4

LuminaryResult store_ARGB8_qoi(const char* filename, const ARGB8* image, const int width, const int height);
LuminaryResult store_as_qoi(
  const char* filename, const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t color_type);
LuminaryResult qoi_encode_RGBA8(const Texture* tex, int* encoded_size, void** data);
LuminaryResult qoi_decode_RGBA8(const void* data, const int size, Texture* texture);

#if __cplusplus
}
#endif

#endif /* LUM_QOI_H */
