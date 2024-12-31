#include "dropdown.h"

#include <stdio.h>

#include "display.h"
#include "windows/subwindow_dropdown.h"

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
  dropdown.hash        = element_compute_hash(args.identifier);

  ElementDropdownData* data = (ElementDropdownData*) &dropdown.data;

  if (args.selected_index < args.num_strings) {
    sprintf(data->text, "%s", args.strings[args.selected_index]);
  }
  else {
    sprintf(data->text, "Paradox ERR");
  }

  ElementMouseResult mouse_result;
  element_apply_context(&dropdown, context, &args.size, display, &mouse_result);

  if (mouse_result.is_clicked && window->external_subwindow) {
    subwindow_dropdown_create(window->external_subwindow, args.selected_index, dropdown.width, dropdown.x, dropdown.y + dropdown.height);

    for (uint32_t string_id = 0; string_id < args.num_strings; string_id++) {
      subwindow_dropdown_add_string(window->external_subwindow, args.strings[string_id]);
    }

    window->state_data = (WindowInteractionStateData){
      .state              = WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED,
      .element_hash       = dropdown.hash,
      .subelement_index   = 0,
      .dropdown_selection = args.selected_index};
  }

  window_push_element(window, &dropdown);

  return mouse_result.is_clicked;
}
