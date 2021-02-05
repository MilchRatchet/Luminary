#include <stdlib.h>
#include <math.h>
#include "Luminary.h"
#include "lib/png.h"
#include "lib/image.h"
#include "lib/cudatest.h"

int main() {
  cudatest();

  const int width  = 3840;
  const int height = 2160;

  const int point = 250;
  const int dist  = 50;

  RGB8* test = (RGB8*)malloc(sizeof(RGB8) * width * height);

  unsigned long ptr = 0;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      test[ptr].r = 255;
      test[ptr].g = 255 * (ptr % 2);
      test[ptr].b = 200;

      ptr++;
    }
  }

  store_as_png(
    "test.png", (uint8_t*) test, sizeof(RGB8) * width * height, width, height,
    PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  return 0;
}
