#ifndef UI_BLIT_H
#define UI_BLIT_H

#include <stdint.h>

#include "UI_structs.h"

void blit_gray(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t value);
void blit_color(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t red, uint8_t green, uint8_t blue);
void blit_color_shaded(uint8_t* dst, int x, int y, int ldd, int hd, int width, int height, uint8_t red, uint8_t green, uint8_t blue);
void blit_UI_internal(UI* ui, uint8_t* target, int width);
void blit_text(UI* ui, SDL_Surface* text, int x, int y, int ldd, int hd);

#endif /* UI_BLIT_H */
