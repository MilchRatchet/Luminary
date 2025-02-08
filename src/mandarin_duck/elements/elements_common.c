#include "elements_common.h"

#include "display.h"
#include "element.h"
#include "hash.h"
#include "ui_renderer_utils.h"
#include "window.h"

void element_apply_context(
  Element* element, WindowContext* context, ElementSize* size, const MouseState* mouse_state, ElementMouseResult* mouse_result) {
  if (size->width != ELEMENT_SIZE_INVALID) {
    element->width = size->width;
  }
  else {
    const float rel_width = fminf(1.0f, fmaxf(0.0f, size->rel_width));

    element->width = (uint32_t) ((context->width - 2 * context->padding) * rel_width);

    // If the element has a relative size, we clamp it to fit into the bounds of the context.
    if (context->is_horizontal) {
      element->width = (context->fill + element->width + 2 * context->padding <= context->width)
                         ? element->width
                         : context->width - context->fill - 2 * context->padding;
    }
  }

  if (size->height != ELEMENT_SIZE_INVALID) {
    element->height = size->height;
  }
  else {
    const float rel_height = fminf(1.0f, fmaxf(0.0f, size->rel_height));

    element->height = (uint32_t) ((context->height - 2 * context->padding) * rel_height);

    // If the element has a relative size, we clamp it to fit into the bounds of the context.
    if (!context->is_horizontal) {
      element->height = (context->fill + element->height + 2 * context->padding <= context->height)
                          ? element->height
                          : context->height - context->fill - 2 * context->padding;
    }
  }

  element->x = context->x + context->padding + ((context->is_horizontal) ? context->fill : 0);
  element->y = context->y + context->padding + ((context->is_horizontal) ? 0 : context->fill);

  context->fill += (context->is_horizontal) ? element->width : element->height;

  const uint32_t mouse_x = mouse_state->x;
  const uint32_t mouse_y = mouse_state->y;

  const bool in_horizontal_bounds = (mouse_x >= element->x) && (mouse_x <= (element->x + element->width));
  const bool in_vertical_bounds   = (mouse_y >= element->y) && (mouse_y <= (element->y + element->height));

  mouse_result->is_hovered = in_horizontal_bounds && in_vertical_bounds;
  mouse_result->is_down    = mouse_result->is_hovered && mouse_state->down;
  mouse_result->is_pressed = mouse_result->is_hovered && (mouse_state->phase == MOUSE_PHASE_PRESSED);
  mouse_result->is_clicked = mouse_result->is_hovered && (mouse_state->phase == MOUSE_PHASE_RELEASED);

  if (mouse_result->is_clicked || mouse_result->is_pressed || mouse_result->is_down) {
    mouse_result->click_rel_x = ((float) (mouse_x - element->x)) / element->width;
    mouse_result->click_rel_y = ((float) (mouse_y - element->y)) / element->height;
  }
}

uint64_t element_compute_hash(const char* context_indentifier, const char* identifier) {
  Hash hash;
  hash_init(&hash);

  hash_string(&hash, context_indentifier);
  hash_string(&hash, identifier);

  return hash.hash;
}
