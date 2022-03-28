#include "UI_text.h"

#include <stdio.h>

#include "log.h"

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void init_text(UI* ui) {
  TTF_Init();
  ui->font = TTF_OpenFont(TTF_FONT_LOCATION, 20);

  if (!ui->font) {
    crash_message(
      "LuminaryFont.ttf was not found. Make sure it resides in the same folder as the "
      "executable!");
  }
}

SDL_Surface* render_text(UI* ui, const char* text) {
  SDL_Color color1 = {255, 255, 255};
  SDL_Color color2 = {0, 0, 0};
  return TTF_RenderText_Shaded(ui->font, text, color1, color2);
}
