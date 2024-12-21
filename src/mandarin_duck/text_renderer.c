#include "text_renderer.h"

// TODO: Let Mandarin Duck have its own embedded files.
void ceb_access(const char* restrict name, void** restrict ptr, int64_t* restrict lmem, uint64_t* restrict info);

#define TTF_FONT_LOCATION "LuminaryFont.ttf"

void text_renderer_create(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  LUM_FAILURE_HANDLE(host_malloc(text_renderer, sizeof(TextRenderer)));

  char* font_data;
  int64_t font_size;
  uint64_t info;

  ceb_access(TTF_FONT_LOCATION, (void**) &font_data, &font_size, &info);

  if (info || !font_size) {
    crash_message("LuminaryFont.ttf was not part of the embedded files. Luminary needs to be rebuild. Ceb Error Code: %zu", info);
  }

  SDL_IOStream* iostream = SDL_IOFromConstMem((const void*) font_data, font_size);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetPointerProperty(sdl_properties, TTF_PROP_FONT_CREATE_IOSTREAM_POINTER, iostream);
  SDL_SetBooleanProperty(sdl_properties, TTF_PROP_FONT_CREATE_IOSTREAM_AUTOCLOSE_BOOLEAN, true);
  SDL_SetNumberProperty(sdl_properties, TTF_PROP_FONT_CREATE_VERTICAL_DPI_NUMBER, 72);

  (*text_renderer)->font = TTF_OpenFontWithProperties(sdl_properties);

  if ((*text_renderer)->font == (TTF_Font*) 0) {
    crash_message("Failed to load font.");
  }
}

void text_renderer_destroy(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  TTF_CloseFont((*text_renderer)->font);

  LUM_FAILURE_HANDLE(host_free(text_renderer));
}
