#include "elements_common.h"

#include "display.h"
#include "element.h"
#include "ui_renderer_utils.h"
#include "window.h"

void element_apply_context(
  Element* element, WindowContext* context, ElementSize* size, Display* display, ElementMouseResult* mouse_result) {
  if (size->is_relative) {
    const float rel_width  = fminf(1.0f, fmaxf(0.0f, size->rel_width));
    const float rel_height = fminf(1.0f, fmaxf(0.0f, size->rel_height));

    element->width  = (uint32_t) ((context->width - 2 * context->padding) * rel_width);
    element->height = (uint32_t) ((context->height - 2 * context->padding) * rel_height);

    // If the element has a relative size, we clamp it to fit into the bounds of the context.
    if (context->is_horizontal) {
      element->width = (context->fill + element->width + 2 * context->padding <= context->width)
                         ? element->width
                         : context->width - context->fill - 2 * context->padding;
    }
    else {
      element->height = (context->fill + element->height + 2 * context->padding <= context->height)
                          ? element->height
                          : context->height - context->fill - 2 * context->padding;
    }
  }
  else {
    element->width  = size->width;
    element->height = size->height;
  }

  element->x = context->x + context->padding + ((context->is_horizontal) ? context->fill : 0);
  element->y = context->y + context->padding + ((context->is_horizontal) ? 0 : context->fill);

  context->fill += (context->is_horizontal) ? element->width : element->height;

  const MouseState* mouse_state = display->mouse_state;

  const uint32_t mouse_x = mouse_state->x;
  const uint32_t mouse_y = mouse_state->y;

  const bool in_horizontal_bounds = (mouse_x >= element->x) && (mouse_x <= (element->x + element->width));
  const bool in_vertical_bounds   = (mouse_y >= element->y) && (mouse_y <= (element->y + element->height));

  mouse_result->is_hovered = in_horizontal_bounds && in_vertical_bounds;
  mouse_result->is_pressed = mouse_result->is_hovered && mouse_state->down;
  mouse_result->is_clicked = mouse_result->is_hovered && (mouse_state->phase == MOUSE_PHASE_RELEASED);
}

uint64_t element_compute_hash(const char* identifier) {
  if (identifier == (const char*) 0)
    return 0;

  size_t string_length = strlen(identifier);

  uint64_t hash = 0;

  while (string_length >= 8) {
    hash = element_hash_accumulate(hash, *(uint64_t*) identifier);

    identifier    = identifier + 8;
    string_length = string_length - 8;
  }

  if (string_length > 0) {
    uint64_t value = 0;

    while (string_length > 0) {
      value |= ((uint64_t) (*identifier)) << (8 * string_length);

      identifier    = identifier + 1;
      string_length = string_length - 1;
    }

    hash = element_hash_accumulate(hash, value);
  }

  return hash;
}
