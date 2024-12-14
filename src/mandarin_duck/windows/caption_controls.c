#include "caption_controls.h"

#include "button.h"
#include "container.h"

static void _window_caption_controls_setup(Window* window) {
  MD_CHECK_NULL_ARGUMENT(window);

  element_container_create(&window->element_container);

  ElementContainerArgs container_args;
  container_args.target_width  = 1.0f;
  container_args.target_height = 1.0f;
  container_args.padding       = window->padding;
  container_args.is_horizontal = window->is_horizontal;

  element_container_set(window->element_container, &container_args);

  {
    ElementButtonArgs button_args;
    button_args.shape = ELEMENT_BUTTON_SHAPE_CIRCLE;
    button_args.color = 0xFFFFFFFF;
  }
}

void window_caption_controls_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  LUM_FAILURE_HANDLE(host_malloc(window, sizeof(Window)));

  _window_caption_controls_setup(*window);
}
