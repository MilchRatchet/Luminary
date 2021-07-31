#include <stdio.h>
#include "UI_text.h"

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void init_text(UI* ui) {
  TTF_Init();
  ui->font = TTF_OpenFont(TTF_FONT_LOCATION, 20);

  if (!ui->font) {
    printf(
      "LuminaryFont.ttf was not found. Make sure it resides in the same folder as the "
      "executable!\n");
    system("Pause");
    exit(1);
  }
}

SDL_Surface* render_text(UI* ui, const char* text) {
  SDL_Color color1 = {255, 255, 255};
  SDL_Color color2 = {0, 0, 0};
  return TTF_RenderText_Shaded(ui->font, text, color1, color2);
}

void blit_text(UI* ui, SDL_Surface* text, int x, int y) {
  uint8_t* text_pixels = (uint8_t*) text->pixels;
  for (int i = 0; i < text->h; i++) {
    for (int j = 0; j < 3 * text->w; j++) {
      const int ui_index        = 3 * x + j + (i + y) * UI_WIDTH * 3;
      const float alpha         = text_pixels[j / 3 + i * text->pitch] / 255.0f;
      const uint8_t ui_pixel    = ui->pixels[ui_index];
      ui->pixels[ui_index]      = alpha * 255 + (1.0f - alpha) * ui_pixel;
      ui->pixels_mask[ui_index] = (alpha > 0.0f) ? 0xff : 0;
    }
  }
}
