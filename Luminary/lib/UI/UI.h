#ifndef UI_H
#define UI_H

#include <stdint.h>
#include "SDL_ttf.h"

#define UI_WIDTH 320
#define UI_HEIGHT 600

struct UI {
  int active;
  int x;
  int y;
  TTF_Font* font;
  uint8_t* pixels;
  float alpha;
  void* scratch;
} typedef UI;

UI init_UI();
void render_UI(UI* ui);
void blit_UI(UI* ui, uint8_t* target, int width, int height);
void free_UI(UI* ui);

#endif /* UI_H */
