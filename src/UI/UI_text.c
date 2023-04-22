#include "UI_text.h"

#include <stdio.h>

#include "ceb.h"
#include "log.h"

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void init_text(UI* ui) {
  TTF_Init();

  int64_t font_size = -1;
  uint64_t info;

  ceb_load(TTF_FONT_LOCATION, (void*) 0, &font_size, &info);

  if (info || !font_size) {
    crash_message("LuminaryFont.ttf was not part of the embedded files. Luminary needs to be rebuild. Ceb Error Code: %zu", info);
  }

  char* font_data = malloc(font_size);
  ceb_load(TTF_FONT_LOCATION, font_data, &font_size, &info);

  if (info) {
    crash_message("Something went wrong loading LuminaryFont.ttf. Ceb Error Code: %zu", info);
  }

  SDL_RWops* sdl_font_mem = SDL_RWFromMem(font_data, font_size);

  if (!sdl_font_mem) {
    crash_message("Unknown error when creating the SDL Font Memory RWops.");
  }

  ui->font = TTF_OpenFontRW(sdl_font_mem, 1, 20);

  if (!ui->font) {
    crash_message(
      "LuminaryFont.ttf was not found. Make sure it resides in the same folder as the "
      "executable!");
  }
}

SDL_Surface* render_text(UI* ui, const char* text) {
  SDL_Color color1 = {255, 255, 255, 255};
  SDL_Color color2 = {0, 0, 0, 255};
  return TTF_RenderText_Shaded(ui->font, text, color1, color2);
}
