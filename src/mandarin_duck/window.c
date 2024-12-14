#include "window.h"

#include <string.h>

#include "display.h"

void window_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  LUM_FAILURE_HANDLE(host_malloc(window, sizeof(Window)));
  memset(*window, 0, sizeof(Window));

  LUM_FAILURE_HANDLE(array_create(&(*window)->element_queue, sizeof(Element), 128));
}

void window_handle_input(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LUM_FAILURE_HANDLE(array_clear(window->element_queue));

  window->context_stack_ptr = 0;

  window->context_stack[0].fill          = 0;
  window->context_stack[0].x             = window->x;
  window->context_stack[0].y             = window->y;
  window->context_stack[0].width         = window->width;
  window->context_stack[0].height        = window->height;
  window->context_stack[0].padding       = window->padding;
  window->context_stack[0].is_horizontal = window->is_horizontal;

  window->action_func(window, display, host);
}

void window_margin(Window* window, uint32_t margin) {
  MD_CHECK_NULL_ARGUMENT(window);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  context->fill += margin;
}

void window_push_section(Window* window);
void window_pop_section(Window* window);

void window_render(Window* window, Display* display) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);

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

  LUM_FAILURE_HANDLE(array_destroy(&(*window)->element_queue));

  LUM_FAILURE_HANDLE(host_free(window));
}
