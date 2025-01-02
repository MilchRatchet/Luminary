#include "window.h"

#include <string.h>

#include "display.h"
#include "ui_renderer_blur.h"

static uint64_t __window_depth_counter = 1;

void window_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  LUM_FAILURE_HANDLE(host_malloc(window, sizeof(Window)));
  memset(*window, 0, sizeof(Window));

  LUM_FAILURE_HANDLE(array_create(&(*window)->element_queue, sizeof(Element), 128));

  (*window)->depth        = __window_depth_counter++;
  (*window)->is_subwindow = false;
}

void window_create_subwindow(Window* window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(&window->external_subwindow);
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

bool window_is_mouse_hover(Window* window, Display* display, const MouseState* mouse_state) {
  if (window->is_visible == false)
    return false;

  if (window->external_subwindow && window_is_mouse_hover(window->external_subwindow, display, mouse_state)) {
    return true;
  }

  const int32_t mouse_x = mouse_state->x;
  const int32_t mouse_y = mouse_state->y;

  const bool in_horizontal_bounds = (mouse_x >= window->x) && (mouse_x <= (int32_t) (window->x + window->width));
  const bool in_vertical_bounds   = (mouse_y >= window->y) && (mouse_y <= (int32_t) (window->y + window->height));

  return in_horizontal_bounds && in_vertical_bounds;
}

static void _window_reset_state(Window* window) {
  window->state_data.state              = WINDOW_INTERACTION_STATE_NONE;
  window->state_data.element_hash       = 0;
  window->state_data.subelement_index   = 0;
  window->state_data.dropdown_selection = 0;
  window->state_data.num_characters     = 0;
  memset(window->state_data.string, 0, WINDOW_STATE_STRING_SIZE);
  window->state_data.force_string_mode_exit = false;

  if (window->external_subwindow) {
    window->external_subwindow->is_visible = false;
  }
}

bool window_handle_input(Window* window, Display* display, LuminaryHost* host, MouseState* mouse_state) {
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

  window->element_has_hover = false;

  switch (window->state_data.state) {
    case WINDOW_INTERACTION_STATE_NONE:
    default:
      break;
    case WINDOW_INTERACTION_STATE_DRAG:
      if (mouse_state->down) {
        window->x += mouse_state->x_motion;
        window->y += mouse_state->y_motion;

        if (window->x < 0)
          window->x = 0;

        if (window->x + window->width > display->width)
          window->x = display->width - window->width;

        if (window->y < 0)
          window->y = 0;

        mouse_state_invalidate(mouse_state);
      }
      else {
        _window_reset_state(window);
      }
      break;
    case WINDOW_INTERACTION_STATE_SLIDER:
      if (mouse_state->down == false) {
        _window_reset_state(window);
        display_set_mouse_visible(display, true);
      }
      break;
    case WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED:
      window_handle_input(window->external_subwindow, display, host, mouse_state);
      window->external_subwindow->propagate_parent_func(window->external_subwindow, window);
      break;
    case WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER:
      _window_reset_state(window);
      break;
    case WINDOW_INTERACTION_STATE_STRING:
      if (mouse_state->phase == MOUSE_PHASE_PRESSED) {
        window->state_data.force_string_mode_exit = true;
      }
      break;
  }

  window->context_stack_ptr = 0;

  window->context_stack[0] = (WindowContext){
    .fill          = 0,
    .x             = window->x,
    .y             = window->y,
    .width         = window->width,
    .height        = window->height,
    .padding       = window->padding,
    .is_horizontal = window->is_horizontal};

  const bool elements_received_action = window->action_func(window, display, host, mouse_state);
  const bool is_mouse_hover           = window_is_mouse_hover(window, display, mouse_state);

  switch (window->state_data.state) {
    case WINDOW_INTERACTION_STATE_NONE:
    case WINDOW_INTERACTION_STATE_DRAG:
    case WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER:
    default:
      break;
    case WINDOW_INTERACTION_STATE_SLIDER:
      if (mouse_state->down == true) {
        mouse_state_invalidate(mouse_state);
      }
      break;
    case WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED:
      if (mouse_state->phase == MOUSE_PHASE_RELEASED) {
        _window_reset_state(window);
      }
      break;
    case WINDOW_INTERACTION_STATE_STRING:
      if (window->state_data.force_string_mode_exit) {
        _window_reset_state(window);
      }
      break;
  }

  if (is_mouse_hover && !window->element_has_hover && window->is_movable && window->state_data.state == WINDOW_INTERACTION_STATE_NONE) {
    if (mouse_state->down && !elements_received_action) {
      window->state_data.state = WINDOW_INTERACTION_STATE_DRAG;
    }
  }

  if (is_mouse_hover && mouse_state->down && !window->fixed_depth) {
    window->depth = __window_depth_counter++;

    if (window->is_subwindow == false) {
      mouse_state_invalidate(mouse_state);
    }
  }

  return is_mouse_hover || elements_received_action;
}

void window_push_element(Window* window, Element* element) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(element);

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, element));
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

  const bool render_subwindow =
    (window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED
     || window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER);

  if (render_subwindow) {
    window_render(window->external_subwindow, display);
  }
}

void window_destroy(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(*window);

  if ((*window)->external_subwindow) {
    window_destroy(&(*window)->external_subwindow);
  }

  if ((*window)->background_blur_buffer) {
    LUM_FAILURE_HANDLE(host_free(&(*window)->background_blur_buffer));
  }

  LUM_FAILURE_HANDLE(array_destroy(&(*window)->element_queue));

  LUM_FAILURE_HANDLE(host_free(window));
}
