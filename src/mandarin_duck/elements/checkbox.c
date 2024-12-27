#include "checkbox.h"

#include "display.h"

static void _element_checkbox_render_func(Element* checkbox, Display* display) {
  ElementCheckBoxData* data = (ElementCheckBoxData*) &checkbox->data;

  const uint32_t color = data->data ? 0xFFD4AF37 : 0xFF000000;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, checkbox->width, checkbox->height, checkbox->x, checkbox->y, 0, 0xFF111111, color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);
}

bool element_checkbox(Window* window, Display* display, ElementCheckBoxArgs args) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element checkbox;

  checkbox.type        = ELEMENT_TYPE_CHECK_BOX;
  checkbox.render_func = _element_checkbox_render_func;
  checkbox.hash        = 0;

  ElementCheckBoxData* data = (ElementCheckBoxData*) &checkbox.data;

  data->size = args.size;

  ElementMouseResult mouse_result;
  element_apply_context(&checkbox, context, &args.size, display, &mouse_result);

  data->data = *(bool*) args.data_binding;

  if (mouse_result.is_clicked) {
    data->data                 = !data->data;
    *(bool*) args.data_binding = data->data;
  }

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &checkbox));

  return mouse_result.is_clicked;
}
