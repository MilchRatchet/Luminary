#include "renderer_status.h"

#include <stdio.h>
#include <string.h>

#include "display.h"
#include "elements/text.h"

static void _window_renderer_status_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  uint32_t num_queue_workers;
  LUM_FAILURE_HANDLE(luminary_host_get_num_queue_workers(host, &num_queue_workers));

  for (uint32_t queue_worker_id = 0; queue_worker_id < num_queue_workers; queue_worker_id++) {
    const char* name;
    LUM_FAILURE_HANDLE(luminary_host_get_queue_worker_name(host, queue_worker_id, &name));

    if (name == (const char*) 0)
      continue;

    const char* string;
    LUM_FAILURE_HANDLE(luminary_host_get_queue_worker_string(host, queue_worker_id, &string));

    if (string == (const char*) 0)
      continue;

    window_push_section(window, 24, 0);
    {
      window_margin(window, 8);

      element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 0.15f, .height = 24},
                           .text         = name,
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = true,
                           .auto_size    = false,
                           .is_clickable = false});

      element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 0.65f, .height = 24},
                           .text         = string,
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = false,
                           .auto_size    = false,
                           .is_clickable = false});

      char text[256];

      double time;
      LUM_FAILURE_HANDLE(luminary_host_get_queue_worker_time(host, queue_worker_id, &time));

      sprintf(text, "(%.2fs)", time);

      element_text(
        window, display, mouse_state,
        (ElementTextArgs) {.color        = 0xFFFFFFFF,
                           .size         = (ElementSize) {.rel_width = 0.2f, .height = 24},
                           .text         = text,
                           .center_x     = false,
                           .center_y     = true,
                           .highlighting = false,
                           .cache_text   = false,
                           .auto_size    = false,
                           .is_clickable = false});
    }
    window_pop_section(window);
  }
}

void window_renderer_status_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type            = WINDOW_TYPE_RENDERER_STATUS;
  (*window)->visibility_mask = WINDOW_VISIBILITY_STATUS;
  (*window)->x               = 0;
  (*window)->y               = 0;
  (*window)->width           = 512;
  (*window)->height          = 48;
  (*window)->padding         = 0;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = true;
  (*window)->is_movable      = false;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->auto_align      = true;
  (*window)->margins =
    (WindowMargins) {.margin_bottom = 32, .margin_right = 32, .margin_left = WINDOW_MARGIN_INVALID, .margin_top = WINDOW_MARGIN_INVALID};
  (*window)->action_func = _window_renderer_status_action;
  (*window)->fixed_depth = true;

  window_allocate_memory(*window);
}
