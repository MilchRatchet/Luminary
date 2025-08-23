#include "checkbox.h"

#include "display.h"

static void _element_checkbox_render_func(Element* checkbox, Display* display) {
  ElementCheckBoxData* data = (ElementCheckBoxData*) &checkbox->data;

  const uint32_t background_color = (data->data) ? MD_COLOR_ACCENT_2 : MD_COLOR_BLACK;
  const UIRendererBackgroundMode background_mode =
    (data->data) ? UI_RENDERER_BACKGROUND_MODE_OPAQUE : UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, checkbox->width, checkbox->height, checkbox->x, checkbox->y, 0, MD_COLOR_BORDER, background_color,
    background_mode);

  if (data->data) {
    const uint32_t padding_x = (checkbox->width >> 1) - 1;
    const uint32_t padding_y = (checkbox->width >> 1) + 2;

    text_renderer_render(
      display->text_renderer, display, "\ue5ca", TEXT_RENDERER_FONT_MATERIAL, MD_COLOR_WHITE, checkbox->x + padding_x,
      checkbox->y + padding_y, true, true, true, (uint32_t*) 0);
  }
}

bool element_checkbox(Window* window, Display* display, const MouseState* mouse_state, ElementCheckBoxArgs args) {
  MD_CHECK_NULL_ARGUMENT(window);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element checkbox;

  checkbox.type        = ELEMENT_TYPE_CHECK_BOX;
  checkbox.render_func = _element_checkbox_render_func;
  checkbox.hash        = 0;

  ElementCheckBoxData* data = (ElementCheckBoxData*) &checkbox.data;

  data->size = args.size;

  ElementMouseResult mouse_result;
  element_apply_context(&checkbox, context, &args.size, mouse_state, &mouse_result);

  data->data = *(bool*) args.data_binding;

  if (mouse_result.is_clicked) {
    data->data                 = !data->data;
    *(bool*) args.data_binding = data->data;

    window->status.received_mouse_action |= true;
  }

  if (mouse_result.is_hovered) {
    window->element_has_hover = true;

    display_set_cursor(display, SDL_SYSTEM_CURSOR_POINTER);
    window->status.received_hover |= true;
  }

  window_push_element(window, &checkbox);

  return mouse_result.is_clicked;
}
