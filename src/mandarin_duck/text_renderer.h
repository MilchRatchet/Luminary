#ifndef MANDARIN_DUCK_TEXT_RENDERER_H
#define MANDARIN_DUCK_TEXT_RENDERER_H

#include <SDL3_ttf/SDL_ttf.h>

#include "utils.h"

struct TextRenderer {
  TTF_Font* font_regular;
  TTF_Font* font_bold;
} typedef TextRenderer;

void text_renderer_create(TextRenderer** text_renderer);
void text_renderer_render(TextRenderer* text_renderer, const char* text, SDL_Surface** surface);
void text_renderer_destroy(TextRenderer** text_renderer);

#endif /* MANDARIN_DUCK_TEXT_RENDERER */
