#ifndef MANDARIN_DUCK_TEXT_RENDERER_H
#define MANDARIN_DUCK_TEXT_RENDERER_H

#include <SDL3_ttf/SDL_ttf.h>

#include "utils.h"

struct Display typedef Display;

enum TextRendererFont { TEXT_RENDERER_FONT_REGULAR, TEXT_RENDERER_FONT_BOLD, TEXT_RENDERER_FONT_COUNT } typedef TextRendererFont;

struct TextRenderer {
  TTF_Font* fonts[TEXT_RENDERER_FONT_COUNT];
  TTF_TextEngine* text_engine;
} typedef TextRenderer;

void text_renderer_create(TextRenderer** text_renderer);
void text_renderer_render(
  TextRenderer* text_renderer, Display* display, const char* text, uint32_t font_id, uint32_t x, uint32_t y, bool center_x, bool center_y);
void text_renderer_destroy(TextRenderer** text_renderer);

#endif /* MANDARIN_DUCK_TEXT_RENDERER */
