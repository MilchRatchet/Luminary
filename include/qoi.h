#ifndef LUM_QOI_H
#define LUM_QOI_H

#include <stdint.h>

#include "texture.h"

#define QOI_COLORTYPE_TRUECOLOR 3
#define QOI_COLORTYPE_TRUECOLOR_ALPHA 4

int store_XRGB8_qoi(const char* filename, const XRGB8* image, const int width, const int height);
int store_as_qoi(const char* filename, const uint8_t* image, const uint32_t width, const uint32_t height, const uint8_t color_type);

#endif /* LUM_QOI_H */
