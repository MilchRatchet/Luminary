#include "slider.h"

#include <stdio.h>
#include <string.h>

#include "display.h"

static void _element_slider_render_func(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  char text[256];
  if (data->is_integer) {
    sprintf(text, "%u", data->data_uint);
  }
  else {
    sprintf(text, "%f", data->data_float);
  }

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, text, &surface);

  if (surface->h > (int32_t) slider->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = data->center_x ? (slider->width - surface->w) >> 1 : 0;
  const uint32_t padding_y = data->center_y ? (slider->height - surface->h) >> 1 : 0;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = slider->x + padding_x;
  dst_rect.y = slider->y + padding_y;
  dst_rect.w = surface->w;
  dst_rect.h = surface->h;

  SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

  SDL_DestroySurface(surface);
}

bool element_slider(Window* window, Display* display, ElementSliderArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element text;

  text.type        = ELEMENT_TYPE_TEXT;
  text.render_func = _element_slider_render_func;

  ElementSliderData* data = (ElementSliderData*) &text.data;

  data->is_integer = args.is_integer;
  data->color      = args.color;
  data->size       = args.size;
  data->center_x   = args.center_x;
  data->center_y   = args.center_y;

  if (data->is_integer) {
    data->data_uint = *(uint32_t*) args.data_binding;
  }
  else {
    data->data_float = *(float*) args.data_binding;
  }

  ElementMouseResult mouse_result;
  element_apply_context(&text, context, &args.size, display, &mouse_result);

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &text));

  return mouse_result.is_clicked;
}
