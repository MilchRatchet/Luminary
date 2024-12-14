#include "button.h"

#include "display.h"
#include "ui_renderer_utils.h"
#include "window.h"

static void _element_button_render_func(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  uint32_t color = (data->is_hovered) ? 0xFFFF00FF : data->color;

  test_render_color(display->buffer, button->x, button->y, button->width, button->height, display->ld, color);
}

bool element_button(Window* window, Display* display, ElementButtonArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element button;

  button.type        = ELEMENT_TYPE_BUTTON;
  button.render_func = _element_button_render_func;

  ElementButtonData* data = (ElementButtonData*) &button.data;

  element_compute_size_and_position(&button, context, &args.size);

  data->shape      = args.shape;
  data->color      = args.color;
  data->is_hovered = element_is_mouse_hover(&button, display);

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &button));

  return element_is_clicked(&button, display);
}
