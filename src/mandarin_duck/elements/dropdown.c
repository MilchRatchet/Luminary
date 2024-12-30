#include "dropdown.h"

#include <stdio.h>

#include "display.h"

static void _element_dropdown_render_func(Element* dropdown, Display* display) {
  ElementDropdownData* data = (ElementDropdownData*) &dropdown->data;

  const uint32_t arrow_size        = dropdown->height;
  const uint32_t text_field_length = dropdown->width - arrow_size;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, text_field_length, dropdown->height, dropdown->x + arrow_size, dropdown->y, 0, 0xFF111111, 0xFF000000,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, arrow_size, arrow_size, dropdown->x, dropdown->y, 0, 0xFF111111, 0xFF111111,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, data->text, TEXT_RENDERER_FONT_REGULAR, &surface);

  if (surface->h > (int32_t) dropdown->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = (text_field_length - surface->w) >> 1;
  const uint32_t padding_y = (dropdown->height - surface->h) >> 1;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = dropdown->x + padding_x + arrow_size;
  dst_rect.y = dropdown->y + padding_y;
  dst_rect.w = surface->w;
  dst_rect.h = surface->h;

  SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

  SDL_DestroySurface(surface);
}

bool element_dropdown(Window* window, Display* display, ElementDropdownArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element dropdown;

  dropdown.type        = ELEMENT_TYPE_DROPDOWN;
  dropdown.render_func = _element_dropdown_render_func;
  dropdown.hash        = 0;

  ElementDropdownData* data = (ElementDropdownData*) &dropdown.data;

  sprintf(data->text, "Test");

  ElementMouseResult mouse_result;
  element_apply_context(&dropdown, context, &args.size, display, &mouse_result);

  window_push_element(window, &dropdown);

  return mouse_result.is_clicked;
}
