#include "dropdown.h"

#include <stdio.h>

#include "display.h"
#include "windows/subwindow_dropdown.h"

static void _element_dropdown_render_func(Element* dropdown, Display* display) {
  ElementDropdownData* data = (ElementDropdownData*) &dropdown->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, dropdown->width, dropdown->height, dropdown->x, dropdown->y, 0, 0xFF111111, 0xFF000000,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, data->text, TEXT_RENDERER_FONT_REGULAR, &surface);

  if (surface->h > (int32_t) dropdown->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = (dropdown->width - surface->w) >> 1;
  const uint32_t padding_y = (dropdown->height - surface->h) >> 1;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = dropdown->x + padding_x;
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

  uint32_t selected_index        = *args.selected_index;
  uint32_t cached_selected_index = selected_index;

  const bool external_clicked_window_is_present = (window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED);

  if (external_clicked_window_is_present && (window->state_data.element_hash == dropdown.hash)) {
    cached_selected_index = window->state_data.dropdown_selection;
  }

  bool selection_changed = false;

  if (cached_selected_index != selected_index) {
    selected_index    = cached_selected_index;
    selection_changed = true;

    *args.selected_index = selected_index;
  }

  if (selected_index < args.num_strings) {
    sprintf(data->text, "%s", args.strings[selected_index]);
  }
  else {
    sprintf(data->text, "Paradox ERR");
  }

  ElementMouseResult mouse_result;
  element_apply_context(&dropdown, context, &args.size, display, &mouse_result);

  if (mouse_result.is_clicked && window->external_subwindow && !external_clicked_window_is_present) {
    subwindow_dropdown_create(window->external_subwindow, selected_index, dropdown.width, dropdown.x, dropdown.y + dropdown.height);

    for (uint32_t string_id = 0; string_id < args.num_strings; string_id++) {
      subwindow_dropdown_add_string(window->external_subwindow, args.strings[string_id]);
    }

    window->state_data = (WindowInteractionStateData){
      .state              = WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED,
      .element_hash       = dropdown.hash,
      .subelement_index   = 0,
      .dropdown_selection = selected_index};
  }

  window_push_element(window, &dropdown);

  return selection_changed || mouse_result.is_clicked;
}
