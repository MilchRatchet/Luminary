#include "color.h"

#include "display.h"

static void _element_color_render_func(Element* color, Display* display) {
  ElementColorData* data = (ElementColorData*) &color->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, color->width, color->height, color->x, color->y, 0, MD_COLOR_BORDER, data->color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);
}

bool element_color(Window* window, const MouseState* mouse_state, ElementColorArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element color;

  color.type        = ELEMENT_TYPE_COLOR;
  color.render_func = _element_color_render_func;
  color.hash        = 0;

  ElementColorData* data = (ElementColorData*) &color.data;

  ElementMouseResult mouse_result;
  element_apply_context(&color, context, &args.size, mouse_state, &mouse_result);

  data->color = args.color;

  window->status.received_hover |= mouse_result.is_hovered;
  window->status.received_mouse_action |= mouse_result.is_clicked;

  window_push_element(window, &color);

  return mouse_result.is_clicked;
}
