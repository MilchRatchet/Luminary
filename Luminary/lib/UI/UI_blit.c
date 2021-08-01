#include <string.h>
#include "UI_blit.h"

void blit_fast(uint8_t* src, int lds, uint8_t* dst, int ldd, int width, int height) {
  for (int i = 0; i < height; i++) {
    memcpy(dst + i * 3 * ldd, src + i * 3 * lds, 3 * width);
  }
}

void blit_gray(uint8_t* dst, int x, int y, int ldd, int width, int height, uint8_t value) {
  for (int i = 0; i < height; i++) {
    memset(dst + (y + i) * 3 * ldd + 3 * x, value, 3 * width);
  }
}
