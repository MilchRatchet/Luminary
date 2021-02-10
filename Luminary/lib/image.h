#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

struct RGB8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} typedef RGB8;

/* 16bit depth is not supported yet due to big endianness being required */
/*struct RGB16 {
  uint16_t r;
  uint16_t g;
  uint16_t b;
} typedef RGB16;*/

struct RGBF {
  float r;
  float g;
  float b;
} typedef RGBF;

#endif /* IMAGE_H */
