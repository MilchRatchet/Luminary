#include "window.h"

#include <string.h>

#include "display.h"
#include "ui_renderer_blur.h"

void window_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  LUM_FAILURE_HANDLE(host_malloc(window, sizeof(Window)));
  memset(*window, 0, sizeof(Window));

  LUM_FAILURE_HANDLE(array_create(&(*window)->element_queue, sizeof(Element), 128));
}

void window_allocate_memory(Window* window) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (!window->background)
    return;

  const size_t required_background_blur_buffer_size = window->width * window->height * sizeof(LuminaryARGB8);

  if (required_background_blur_buffer_size <= window->background_blur_buffer_size)
    return;

  if (window->background_blur_buffer) {
    LUM_FAILURE_HANDLE(host_free(&window->background_blur_buffer));
  }

  LUM_FAILURE_HANDLE(host_malloc(&window->background_blur_buffer, required_background_blur_buffer_size));
  window->background_blur_buffer_ld   = window->width * sizeof(LuminaryARGB8);
  window->background_blur_buffer_size = required_background_blur_buffer_size;
}

static bool window_is_mouse_hover(Window* window, Display* display) {
  const MouseState* mouse_state = display->mouse_state;

  const int32_t mouse_x = mouse_state->x;
  const int32_t mouse_y = mouse_state->y;

  const bool in_horizontal_bounds = (mouse_x >= window->x) && (mouse_x <= (int32_t) (window->x + window->width));
  const bool in_vertical_bounds   = (mouse_y >= window->y) && (mouse_y <= (int32_t) (window->y + window->height));

  return in_horizontal_bounds && in_vertical_bounds;
}

bool window_handle_input(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LUM_FAILURE_HANDLE(array_clear(window->element_queue));

  if (window->auto_align) {
    if (window->margins.margin_bottom != WINDOW_MARGIN_INVALID) {
      window->y = display->height - window->margins.margin_bottom - window->height;
    }

    if (window->margins.margin_top != WINDOW_MARGIN_INVALID) {
      window->y = window->margins.margin_top;
    }

    if (window->margins.margin_right != WINDOW_MARGIN_INVALID) {
      window->x = display->width - window->margins.margin_right - window->width;
    }

    if (window->margins.margin_left != WINDOW_MARGIN_INVALID) {
      window->x = window->margins.margin_left;
    }
  }

  window->context_stack_ptr = 0;

  window->context_stack[0].fill          = 0;
  window->context_stack[0].x             = window->x;
  window->context_stack[0].y             = window->y;
  window->context_stack[0].width         = window->width;
  window->context_stack[0].height        = window->height;
  window->context_stack[0].padding       = window->padding;
  window->context_stack[0].is_horizontal = window->is_horizontal;

  window->element_has_hover = false;

  const bool elements_received_action = window->action_func(window, display, host);
  const bool is_mouse_hover           = window_is_mouse_hover(window, display);

  if (elements_received_action) {
    return true;
  }

  if (is_mouse_hover && !window->element_has_hover && window->is_movable && window->state_data.state == WINDOW_INTERACTION_STATE_NONE) {
    if (display->mouse_state->down) {
      window->x += display->mouse_state->x_motion;
      window->y += display->mouse_state->y_motion;

      if (window->x < 0)
        window->x = 0;

      if (window->x + window->width > display->width)
        window->x = display->width - window->width;

      if (window->y < 0)
        window->y = 0;
    }
  }

  switch (window->state_data.state) {
    case WINDOW_INTERACTION_STATE_NONE:
      break;
    case WINDOW_INTERACTION_STATE_SLIDER:
      if (display->mouse_state->down == false) {
        window->state_data.state            = WINDOW_INTERACTION_STATE_NONE;
        window->state_data.element_hash     = 0;
        window->state_data.subelement_index = 0;

        display_set_mouse_visible(display, true);
      }
      break;
    default:
      break;
  }

  return is_mouse_hover;
}

void window_margin(Window* window, uint32_t margin) {
  MD_CHECK_NULL_ARGUMENT(window);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  context->fill += margin;
}

void window_margin_relative(Window* window, float margin) {
  MD_CHECK_NULL_ARGUMENT(window);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  context->fill += (context->is_horizontal ? context->width : context->height) * margin;
}

void window_push_section(Window* window, uint32_t size, uint32_t padding) {
  MD_CHECK_NULL_ARGUMENT(window);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  window->context_stack_ptr++;

  WindowContext* new_context = window->context_stack + window->context_stack_ptr;

  memcpy(new_context, context, sizeof(WindowContext));

  new_context->is_horizontal = !context->is_horizontal;
  new_context->width         = (new_context->is_horizontal) ? context->width - 2 * context->padding : size;
  new_context->height        = (new_context->is_horizontal) ? size : context->height - 2 * context->padding;
  new_context->x             = ((context->is_horizontal) ? context->x + context->fill : context->x) + context->padding;
  new_context->y             = ((context->is_horizontal) ? context->y : context->y + context->fill) + context->padding;
  new_context->padding       = padding;
  new_context->fill          = 0;

  context->fill += size;
}

void window_pop_section(Window* window) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (window->context_stack_ptr == 0) {
    crash_message("No window context to pop.");
  }

  window->context_stack_ptr--;
}

void window_render(Window* window, Display* display) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

  if (window->auto_size) {
    WindowContext* main_context = window->context_stack;

    if (window->is_horizontal) {
      window->width = main_context->fill + 2 * window->padding;
    }
    else {
      window->height = main_context->fill + 2 * window->padding;
    }
  }

  if (window->width == 0 || window->height == 0)
    return;

  // This must be called after potential auto-size work.
  window_allocate_memory(window);

  if (window->background) {
    ui_renderer_create_window_background(display->ui_renderer, display, window);
    ui_renderer_render_window(display->ui_renderer, display, window);
  }

  uint32_t num_elements;
  LUM_FAILURE_HANDLE(array_get_num_elements(window->element_queue, &num_elements));

  for (uint32_t element_id = 0; element_id < num_elements; element_id++) {
    Element* element = window->element_queue + element_id;

    element->render_func(element, display);
  }
}

void window_destroy(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(*window);

  if ((*window)->background_blur_buffer) {
    LUM_FAILURE_HANDLE(host_free(&(*window)->background_blur_buffer));
  }

  LUM_FAILURE_HANDLE(array_destroy(&(*window)->element_queue));

  LUM_FAILURE_HANDLE(host_free(window));
}
