#ifndef UI_BLIT_H
#define UI_BLIT_H

#include <stdint.h>

void blit_fast(uint8_t* src, int lds, uint8_t* dst, int ldd, int width, int height);
void blit_gray(uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t value);

#endif /* UI_BLIT_H */
