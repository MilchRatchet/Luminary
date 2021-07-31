#include <stdio.h>
#include "UI_text.h"

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void init_text(UI* ui) {
  TTF_Init();
  ui->font = TTF_OpenFont(TTF_FONT_LOCATION, 20);

  if (!ui->font)
    printf("NOTFOUND!\n");
}

SDL_Surface* render_text(UI* ui, const char* text) {
  SDL_Color color = {250, 250, 250};
  return TTF_RenderText_Solid(ui->font, text, color);
}

void blit_text(UI* ui, SDL_Surface* text, int x, int y) {
  uint8_t* text_pixels = (uint8_t*) text->pixels;
  for (int i = 0; i < text->h; i++) {
    for (int j = 0; j < 3 * text->w; j++) {
      const int ui_index   = x + j + (i + y) * UI_WIDTH * 3;
      uint8_t text_pixel   = text_pixels[j / 3 + i * text->pitch];
      uint8_t ui_pixel     = ui->pixels[ui_index];
      ui->pixels[ui_index] = (text_pixel) ? 255 : ui_pixel;
    }
  }
}
