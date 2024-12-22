#include "text_renderer.h"

// TODO: Let Mandarin Duck have its own embedded files.
void ceb_access(const char* restrict name, void** restrict ptr, int64_t* restrict lmem, uint64_t* restrict info);

#define TTF_FONT_LOCATION "LuminaryFont.ttf"
#define TTF_FONT_LOCATION_BOLD "LuminaryFontBold.ttf"

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

  (*text_renderer)->font_regular = _text_renderer_load_font(TTF_FONT_LOCATION);
  (*text_renderer)->font_bold    = _text_renderer_load_font(TTF_FONT_LOCATION_BOLD);

  if ((*text_renderer)->font_regular == (TTF_Font*) 0) {
    crash_message("Failed to load regular font.");
  }

  if ((*text_renderer)->font_bold == (TTF_Font*) 0) {
    crash_message("Failed to load bold font.");
  }
}

void text_renderer_render(TextRenderer* text_renderer, const char* text, SDL_Surface** surface) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  SDL_Color opaque      = {0xFF, 0xFF, 0xFF, 0xFF};
  SDL_Color transparent = {0xFF, 0, 0, 0xFF};

  *surface = TTF_RenderText_Blended(text_renderer->font_bold, text, 0, opaque);
}

void text_renderer_destroy(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  TTF_CloseFont((*text_renderer)->font_regular);
  TTF_CloseFont((*text_renderer)->font_bold);

  LUM_FAILURE_HANDLE(host_free(text_renderer));
}
