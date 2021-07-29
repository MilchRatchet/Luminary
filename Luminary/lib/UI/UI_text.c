#include <stdio.h>
#include "UI_text.h"

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void init_text(UI* ui) {
  TTF_Init();
  ui->font = TTF_OpenFont(TTF_FONT_LOCATION, 20);

  if (!ui->font)
    printf("NOTFOUND!\n");
}

SDL_Surface* render_text(UI* ui) {
  SDL_Color color = {250, 250, 250};
  return TTF_RenderText_Solid(ui->font, "General", color);
}
