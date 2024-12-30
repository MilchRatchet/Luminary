#include "color.h"

#include "display.h"

static void _element_color_render_func(Element* color, Display* display) {
  ElementColorData* data = (ElementColorData*) &color->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, color->width, color->height, color->x, color->y, 0, 0xFF111111, data->color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);
}

bool element_color(Window* window, Display* display, ElementColorArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element color;

  color.type        = ELEMENT_TYPE_COLOR;
  color.render_func = _element_color_render_func;
  color.hash        = 0;

  ElementColorData* data = (ElementColorData*) &color.data;

  ElementMouseResult mouse_result;
  element_apply_context(&color, context, &args.size, display, &mouse_result);

  data->color = args.color;

  window_push_element(window, &color);

  return mouse_result.is_clicked;
}
