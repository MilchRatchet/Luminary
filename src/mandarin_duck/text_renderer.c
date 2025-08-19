#include "text_renderer.h"

#include "display.h"
#include "hash.h"

// TODO: Let Mandarin Duck have its own embedded files.
void ceb_access(const char* restrict name, void** restrict ptr, int64_t* restrict lmem, uint64_t* restrict info);

// We used to use TTF_SetFontSDF(..., true) which caused the text to extrude by 8 pixels.
// See https://github.com/libsdl-org/SDL_ttf/issues/521
#define TEXT_PADDING (8)

static const char* font_file_locations[TEXT_RENDERER_FONT_COUNT] = {
  [TEXT_RENDERER_FONT_REGULAR]  = "LuminaryFont.ttf",
  [TEXT_RENDERER_FONT_BOLD]     = "LuminaryFontBold.ttf",
  [TEXT_RENDERER_FONT_MATERIAL] = "MaterialSymbols.ttf"};

static const float font_sizes[TEXT_RENDERER_FONT_COUNT] =
  {[TEXT_RENDERER_FONT_REGULAR] = 15.0f, [TEXT_RENDERER_FONT_BOLD] = 15.0f, [TEXT_RENDERER_FONT_MATERIAL] = 25.0f};

static const int32_t font_offset_x[TEXT_RENDERER_FONT_COUNT] =
  {[TEXT_RENDERER_FONT_REGULAR] = 1, [TEXT_RENDERER_FONT_BOLD] = 1, [TEXT_RENDERER_FONT_MATERIAL] = 0};

static const int32_t font_offset_y[TEXT_RENDERER_FONT_COUNT] =
  {[TEXT_RENDERER_FONT_REGULAR] = 1, [TEXT_RENDERER_FONT_BOLD] = 1, [TEXT_RENDERER_FONT_MATERIAL] = -2};

static TTF_Font* _text_renderer_load_font(const char* font_location, const float size) {
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
  SDL_SetFloatProperty(sdl_properties, TTF_PROP_FONT_CREATE_SIZE_FLOAT, size);
  SDL_SetNumberProperty(sdl_properties, TTF_PROP_FONT_CREATE_VERTICAL_DPI_NUMBER, 72);

  return TTF_OpenFontWithProperties(sdl_properties);
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
    (*text_renderer)->fonts[font_id] = _text_renderer_load_font(font_file_locations[font_id], font_sizes[font_id]);

    if ((*text_renderer)->fonts[font_id] == (TTF_Font*) 0) {
      crash_message("Failed to create font: %s.", font_file_locations[font_id]);
    }

    TTF_SetFontHinting((*text_renderer)->fonts[font_id], TTF_HINTING_LIGHT_SUBPIXEL);
  }
}

static TTF_Text* _text_renderer_acquire_text_instance(
  TextRenderer* text_renderer, const char* text, uint32_t font_id, const bool use_cache, bool* loaded_from_cache_ptr, size_t* hash_ptr) {
  TTF_Text* text_instance = (TTF_Text*) 0;
  bool loaded_from_cache  = false;

  Hash hash;
  hash_init(&hash);

  if (use_cache) {
    hash_string(&hash, text);

    uint32_t entry = hash.hash & TEXT_RENDERER_CACHE_SIZE_MASK;

    uint32_t offset = 0;
    while (text_renderer->cache.entries[(entry + offset) & TEXT_RENDERER_CACHE_SIZE_MASK].hash != hash.hash
           && offset < TEXT_RENDERER_CACHE_SIZE) {
      offset++;
    }

    entry = (entry + offset) & TEXT_RENDERER_CACHE_SIZE_MASK;

    if (text_renderer->cache.entries[entry].hash == hash.hash) {
      text_instance     = text_renderer->cache.entries[entry].text;
      loaded_from_cache = true;
    }
  }

  if (!loaded_from_cache) {
    text_instance = TTF_CreateText(text_renderer->text_engine, text_renderer->fonts[font_id], text, 0);
  }

  *loaded_from_cache_ptr = loaded_from_cache;
  *hash_ptr              = hash.hash;

  return text_instance;
}

static void _text_renderer_release_text_instance(
  TextRenderer* text_renderer, TTF_Text* text_instance, size_t hash, bool use_cache, bool loaded_from_cache) {
  bool destroy_text = !loaded_from_cache;

  if (!loaded_from_cache && use_cache) {
    uint32_t entry = hash & TEXT_RENDERER_CACHE_SIZE_MASK;

    uint32_t offset = 0;
    for (; text_renderer->cache.entries[(entry + offset) & TEXT_RENDERER_CACHE_SIZE_MASK].text != (TTF_Text*) 0
           && offset < TEXT_RENDERER_CACHE_SIZE;
         offset++) {
    }

    entry = (entry + offset) & TEXT_RENDERER_CACHE_SIZE_MASK;

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
}

void text_renderer_render(
  TextRenderer* text_renderer, Display* display, const char* text, uint32_t font_id, uint32_t color, uint32_t x, uint32_t y, bool center_x,
  bool center_y, bool use_cache, uint32_t* text_width) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(text);

  bool loaded_from_cache = false;
  size_t hash            = 0;

  TTF_Text* text_instance = _text_renderer_acquire_text_instance(text_renderer, text, font_id, use_cache, &loaded_from_cache, &hash);

  int32_t width;
  int32_t height;
  TTF_GetTextSize(text_instance, &width, &height);

  TTF_SetTextColor(text_instance, (color >> 16) & 0xFF, (color >> 8) & 0xFF, (color >> 0) & 0xFF, (color >> 24) & 0xFF);

  if (center_x) {
    x = x - (width >> 1);
    x += font_offset_x[font_id];
  }

  if (center_y) {
    y = y - (height >> 1);
    y += font_offset_y[font_id];
  }

  if (text_width) {
    width += 2 * TEXT_PADDING;
    *text_width = (uint32_t) width;
  }

  TTF_DrawSurfaceText(text_instance, x, y, display->sdl_surface);

  _text_renderer_release_text_instance(text_renderer, text_instance, hash, use_cache, loaded_from_cache);

  // For some reason, the text sometimes has 0 opacity so we need to overwrite the opacity here
  int32_t blit_width  = ((x + width) <= display->width) ? width : display->width - x;
  int32_t blit_height = ((y + height) <= display->height) ? height : display->height - y;

  uint8_t* dst = display->buffer;

  for (int32_t y_offset = 0; y_offset < blit_height; y_offset++) {
    for (int32_t x_offset = 0; x_offset < blit_width; x_offset++) {
      dst[(x + x_offset) * 4 + (y + y_offset) * display->pitch + 3] = 0xFF;
    }
  }
}

void text_renderer_compute_size(
  TextRenderer* text_renderer, const char* text, uint32_t font_id, bool use_cache, uint32_t* text_width, uint32_t* text_height) {
  MD_CHECK_NULL_ARGUMENT(text_renderer);
  MD_CHECK_NULL_ARGUMENT(text);

  bool loaded_from_cache = false;
  size_t hash            = 0;

  TTF_Text* text_instance = _text_renderer_acquire_text_instance(text_renderer, text, font_id, use_cache, &loaded_from_cache, &hash);

  int32_t width;
  int32_t height;
  TTF_GetTextSize(text_instance, &width, &height);

  width += 2 * TEXT_PADDING;
  height += 2 * TEXT_PADDING;

  if (text_width) {
    *text_width = (uint32_t) width;
  }

  if (text_height) {
    *text_height = (uint32_t) height;
  }

  _text_renderer_release_text_instance(text_renderer, text_instance, hash, use_cache, loaded_from_cache);
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
