#include "text.h"

#include "display.h"
#include "text_renderer.h"

static void _element_text_render_func(Element* text, Display* display) {
  ElementTextData* data = (ElementTextData*) &text->data;

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, data->text, TEXT_RENDERER_FONT_BOLD, &surface);

  if (surface->h > (int32_t) text->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = data->center_x ? (text->width - surface->w) >> 1 : 0;
  const uint32_t padding_y = data->center_y ? (text->height - surface->h) >> 1 : 0;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = text->x + padding_x;
  dst_rect.y = text->y + padding_y;
  dst_rect.w = surface->w;
  dst_rect.h = surface->h;

  SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

  SDL_DestroySurface(surface);
}

bool element_text(Window* window, Display* display, ElementTextArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element text;

  text.type        = ELEMENT_TYPE_TEXT;
  text.render_func = _element_text_render_func;
  text.hash        = 0;

  ElementTextData* data = (ElementTextData*) &text.data;

  data->color    = args.color;
  data->size     = args.size;
  data->center_x = args.center_x;
  data->center_y = args.center_y;

  const size_t text_size = strlen(args.text);

  if (text_size > 256) {
    crash_message("Text is too long.");
  }

  memcpy(data->text, args.text, text_size);
  memset(data->text + text_size, 0, 256 - text_size);

  ElementMouseResult mouse_result;
  element_apply_context(&text, context, &args.size, display, &mouse_result);

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &text));

  return mouse_result.is_clicked;
}
