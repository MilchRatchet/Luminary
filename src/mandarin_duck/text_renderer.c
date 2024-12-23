#include "text_renderer.h"

// TODO: Let Mandarin Duck have its own embedded files.
void ceb_access(const char* restrict name, void** restrict ptr, int64_t* restrict lmem, uint64_t* restrict info);

static const char* font_file_locations[TEXT_RENDERER_FONT_COUNT] =
  {[TEXT_RENDERER_FONT_REGULAR] = "LuminaryFont.ttf", [TEXT_RENDERER_FONT_BOLD] = "LuminaryFontBold.ttf"};

static TTF_Font* _text_renderer_load_font(const char* font_location) {
  char* font_data;
  int64_t font_size;
  uint64_t info;

  ceb_access(font_location, (void**) &font_data, &font_size, &info);

  if (info || !font_size) {
    crash_message("%s was not part of the embedded files. Luminary needs to be rebuild. Ceb Error Code: %zu", font_location, info);
  }

  SDL_IOStream* iostream = SDL_IOFromConstMem((const void*) font_data, font_size);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetPointerProperty(sdl_properties, TTF_PROP_FONT_CREATE_IOSTREAM_POINTER, iostream);
  SDL_SetBooleanProperty(sdl_properties, TTF_PROP_FONT_CREATE_IOSTREAM_AUTOCLOSE_BOOLEAN, true);
  SDL_SetFloatProperty(sdl_properties, TTF_PROP_FONT_CREATE_SIZE_FLOAT, 15.0f);
  SDL_SetNumberProperty(sdl_properties, TTF_PROP_FONT_CREATE_VERTICAL_DPI_NUMBER, 72);

  return TTF_OpenFontWithProperties(sdl_properties);
}

void text_renderer_create(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  LUM_FAILURE_HANDLE(host_malloc(text_renderer, sizeof(TextRenderer)));

  for (uint32_t font_id = 0; font_id < TEXT_RENDERER_FONT_COUNT; font_id++) {
    (*text_renderer)->fonts[font_id] = _text_renderer_load_font(font_file_locations[font_id]);

    if ((*text_renderer)->fonts[font_id] == (TTF_Font*) 0) {
      crash_message("Failed to font: %s.", font_file_locations[font_id]);
    }
  }
}

void text_renderer_render(TextRenderer* text_renderer, const char* text, uint32_t font_id, SDL_Surface** surface) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  SDL_Color opaque      = {0xFF, 0xFF, 0xFF, 0xFF};
  SDL_Color transparent = {0xFF, 0, 0, 0xFF};

  *surface = TTF_RenderText_Blended(text_renderer->fonts[font_id], text, 0, opaque);
}

void text_renderer_destroy(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  for (uint32_t font_id = 0; font_id < TEXT_RENDERER_FONT_COUNT; font_id++) {
    TTF_CloseFont((*text_renderer)->fonts[font_id]);
  }

  LUM_FAILURE_HANDLE(host_free(text_renderer));
}
