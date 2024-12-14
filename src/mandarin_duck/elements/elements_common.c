#include "elements_common.h"

#include "display.h"
#include "element.h"
#include "window.h"

void element_compute_size_and_position(Element* element, WindowContext* context, ElementSize* size) {
  if (size->is_relative) {
    element->width  = (uint32_t) ((context->width - 2 * context->padding) * size->rel_width);
    element->height = (uint32_t) ((context->height - 2 * context->padding) * size->rel_height);
  }
  else {
    element->width  = size->width;
    element->height = size->height;
  }

  element->x = context->x + context->padding + ((context->is_horizontal) ? context->fill : 0);
  element->y = context->y + context->padding + ((context->is_horizontal) ? 0 : context->fill);

  context->fill += (context->is_horizontal) ? element->width : element->height;
}

bool element_is_mouse_hover(Element* element, Display* display) {
  const MouseState* mouse_state = display->mouse_state;

  const uint32_t mouse_x = mouse_state->x;
  const uint32_t mouse_y = mouse_state->y;

  const bool in_horizontal_bounds = (mouse_x >= element->x) && (mouse_x <= (element->x + element->width));
  const bool in_vertical_bounds   = (mouse_y >= element->y) && (mouse_y <= (element->y + element->height));

  return in_horizontal_bounds && in_vertical_bounds;
}

bool element_is_pressed(Element* element, Display* display) {
  const MouseState* mouse_state = display->mouse_state;

  return element_is_mouse_hover(element, display) && mouse_state->down;
}

bool element_is_clicked(Element* element, Display* display) {
  const MouseState* mouse_state = display->mouse_state;

  return element_is_mouse_hover(element, display) && (mouse_state->phase == MOUSE_PHASE_RELEASED);
}
