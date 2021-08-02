#ifndef UI_BLIT_H
#define UI_BLIT_H

#include <stdint.h>
#include "UI_structs.h"

void blit_fast(uint8_t* src, int lds, uint8_t* dst, int ldd, int width, int height);
void blit_gray(uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t value);
void blit_color(
  uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t red, uint8_t green,
  uint8_t blue);
void blit_color_shaded(
  uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t red, uint8_t green,
  uint8_t blue);
void blit_UI_internal(UI* ui, uint8_t* target, int width, int height);

#endif /* UI_BLIT_H */
