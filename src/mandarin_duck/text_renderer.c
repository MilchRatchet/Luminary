#include "text_renderer.h"

#include "display.h"

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

#define FNV_PRIME (0x01000193u)
#define FNV_OFFSET_BASIS (0x811c9dc5u)

static size_t _text_renderer_text_hash(const char* string) {
  const size_t string_length = strlen(string);

  uint32_t hash = FNV_OFFSET_BASIS;

  for (size_t offset = 0; offset < string_length; offset++) {
    hash ^= (size_t) string[offset];
    hash *= FNV_PRIME;
  }

  return hash;
}

void text_renderer_create(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  LUM_FAILURE_HANDLE(host_malloc(text_renderer, sizeof(TextRenderer)));
  memset(*text_renderer, 0, sizeof(TextRenderer));

  (*text_renderer)->text_engine = TTF_CreateSurfaceTextEngine();

  if ((*text_renderer)->text_engine == (TTF_TextEngine*) 0) {
    crash_message("Failed to create text engine.");
  }

  for (uint32_t font_id = 0; font_id < TEXT_RENDERER_FONT_COUNT; font_id++) {
    (*text_renderer)->fonts[font_id] = _text_renderer_load_font(font_file_locations[font_id]);

    if ((*text_renderer)->fonts[font_id] == (TTF_Font*) 0) {
      crash_message("Failed to create font: %s.", font_file_locations[font_id]);
    }

    TTF_SetFontHinting((*text_renderer)->fonts[font_id], TTF_HINTING_LIGHT_SUBPIXEL);
    TTF_SetFontSDF((*text_renderer)->fonts[font_id], true);
  }
}

void text_renderer_render(
  TextRenderer* text_renderer, Display* display, const char* text, uint32_t font_id, uint32_t x, uint32_t y, bool center_x, bool center_y,
  bool use_cache) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(text);

  TTF_Text* text_instance = (TTF_Text*) 0;
  bool loaded_from_cache  = false;
  size_t hash             = 0;

  if (use_cache) {
    hash = _text_renderer_text_hash(text);

    uint32_t entry = hash & TEXT_RENDERER_CACHE_SIZE_MASK;

    uint32_t offset = 0;
    for (; text_renderer->cache.entries[entry + offset].hash != hash && offset < TEXT_RENDERER_CACHE_SIZE; offset++) {
    }

    entry += offset;

    if (text_renderer->cache.entries[entry].hash == hash) {
      text_instance     = text_renderer->cache.entries[entry].text;
      loaded_from_cache = true;
    }
  }

  if (!loaded_from_cache) {
    text_instance = TTF_CreateText(text_renderer->text_engine, text_renderer->fonts[font_id], text, 0);
  }

  int32_t width;
  int32_t height;
  TTF_GetTextSize(text_instance, &width, &height);

  if (center_x) {
    x = x - (width >> 1);
  }

  if (center_y) {
    y = y - (height >> 1);
  }

  TTF_DrawSurfaceText(text_instance, x, y, display->sdl_surface);

  bool destroy_text = !loaded_from_cache;

  if (!loaded_from_cache && use_cache) {
    uint32_t entry = hash & TEXT_RENDERER_CACHE_SIZE_MASK;

    uint32_t offset = 0;
    for (; text_renderer->cache.entries[entry].text != (TTF_Text*) 0 && offset < TEXT_RENDERER_CACHE_SIZE; offset++) {
    }

    entry += offset;

    if (text_renderer->cache.entries[entry].text == (TTF_Text*) 0) {
      text_renderer->cache.num_entries++;
      text_renderer->cache.entries[entry].hash = hash;
      text_renderer->cache.entries[entry].text = text_instance;
      destroy_text                             = false;

      if (text_renderer->cache.num_entries > (TEXT_RENDERER_CACHE_SIZE >> 1)) {
        warn_message("Text renderer cache is getting full, are temporary strings cached?");
      }
    }
  }

  if (destroy_text) {
    TTF_DestroyText(text_instance);
  }

  // For some reason, the text sometimes has 0 opacity so we need to overwrite the opacity here
  int32_t blit_width  = ((x + width) <= display->width) ? width : display->width - x;
  int32_t blit_height = ((y + height) <= display->height) ? height : display->height - y;

  uint8_t* dst = display->buffer;

  for (int32_t y_offset = 0; y_offset < blit_height; y_offset++) {
    for (int32_t x_offset = 0; x_offset < blit_width; x_offset++) {
      dst[(x + x_offset) * 4 + (y + y_offset) * display->ld + 3] = 0xFF;
    }
  }
}

void text_renderer_destroy(TextRenderer** text_renderer) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);

  for (uint32_t entry = 0; entry < TEXT_RENDERER_CACHE_SIZE; entry++) {
    if ((*text_renderer)->cache.entries[entry].text != (TTF_Text*) 0) {
      TTF_DestroyText((*text_renderer)->cache.entries[entry].text);
    }
  }

  TTF_DestroySurfaceTextEngine((*text_renderer)->text_engine);

  for (uint32_t font_id = 0; font_id < TEXT_RENDERER_FONT_COUNT; font_id++) {
    TTF_CloseFont((*text_renderer)->fonts[font_id]);
  }

  LUM_FAILURE_HANDLE(host_free(text_renderer));
}
