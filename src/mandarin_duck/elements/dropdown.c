#include "dropdown.h"

#include <stdio.h>

#include "display.h"
#include "windows/subwindow_dropdown.h"

static void _element_dropdown_render_func(Element* dropdown, Display* display) {
  ElementDropdownData* data = (ElementDropdownData*) &dropdown->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, dropdown->width, dropdown->height, dropdown->x, dropdown->y, 0, 0xFF111111, 0xFF000000,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  const uint32_t padding_x = dropdown->width >> 1;
  const uint32_t padding_y = dropdown->height >> 1;

  text_renderer_render(
    display->text_renderer, display, data->text, TEXT_RENDERER_FONT_REGULAR, dropdown->x + padding_x, dropdown->y + padding_y, true, true,
    true, (uint32_t*) 0);
}

bool element_dropdown(Window* window, const MouseState* mouse_state, ElementDropdownArgs args) {
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
  element_apply_context(&dropdown, context, &args.size, mouse_state, &mouse_result);

  if (mouse_result.is_pressed && window->external_subwindow && !external_clicked_window_is_present) {
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
