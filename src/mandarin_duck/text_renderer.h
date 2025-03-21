#ifndef MANDARIN_DUCK_TEXT_RENDERER_H
#define MANDARIN_DUCK_TEXT_RENDERER_H

#include <SDL3_ttf/SDL_ttf.h>

#include "utils.h"

struct Display typedef Display;

enum TextRendererFont {
  TEXT_RENDERER_FONT_REGULAR,
  TEXT_RENDERER_FONT_BOLD,
  TEXT_RENDERER_FONT_MATERIAL,
  TEXT_RENDERER_FONT_COUNT
} typedef TextRendererFont;

#define TEXT_RENDERER_CACHE_SIZE 1024
#define TEXT_RENDERER_CACHE_SIZE_MASK (TEXT_RENDERER_CACHE_SIZE - 1)

struct TextRendererCacheEntry {
  size_t hash;
  TTF_Text* text;
} typedef TextRendererCacheEntry;

struct TextRendererCache {
  TextRendererCacheEntry entries[1024];
  uint32_t num_entries;
} typedef TextRendererCache;

struct TextRenderer {
  TTF_Font* fonts[TEXT_RENDERER_FONT_COUNT];
  TTF_TextEngine* text_engine;
  TextRendererCache cache;
} typedef TextRenderer;

void text_renderer_create(TextRenderer** text_renderer);
void text_renderer_render(
  TextRenderer* text_renderer, Display* display, const char* text, uint32_t font_id, uint32_t color, uint32_t x, uint32_t y, bool center_x,
  bool center_y, bool use_cache, uint32_t* text_width);
void text_renderer_compute_size(
  TextRenderer* text_renderer, const char* text, uint32_t font_id, bool use_cache, uint32_t* text_width, uint32_t* text_height);
void text_renderer_destroy(TextRenderer** text_renderer);

#endif /* MANDARIN_DUCK_TEXT_RENDERER */
