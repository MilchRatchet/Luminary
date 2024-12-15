#include "button.h"

#include "display.h"
#include "ui_renderer_utils.h"
#include "window.h"

static void _element_button_render_func(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  uint32_t color = (data->is_pressed) ? data->press_color : ((data->is_hovered) ? data->hover_color : data->color);

  test_render_color(
    display->buffer, button->x, button->y, button->width, button->height, display->ld, color, display->ui_renderer->disk_mask);
}

bool element_button(Window* window, Display* display, ElementButtonArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element button;

  button.type        = ELEMENT_TYPE_BUTTON;
  button.render_func = _element_button_render_func;

  ElementButtonData* data = (ElementButtonData*) &button.data;

  ElementMouseResult mouse_result;
  element_apply_context(&button, context, &args.size, display, &mouse_result);

  data->shape       = args.shape;
  data->color       = args.color;
  data->hover_color = args.hover_color;
  data->press_color = args.press_color;
  data->is_hovered  = mouse_result.is_hovered;
  data->is_pressed  = mouse_result.is_pressed;

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &button));

  return mouse_result.is_clicked;
}
