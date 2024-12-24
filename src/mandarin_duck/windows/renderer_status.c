#include "renderer_status.h"

#include <stdio.h>
#include <string.h>

#include "display.h"
#include "elements/text.h"

static bool _window_renderer_status_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  const char* string;
  LUM_FAILURE_HANDLE(luminary_host_get_queue_string(host, &string));

  if (string != (const char*) 0) {
    window_push_section(window, 24, 0);
    {
      window_margin(window, 8);

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.2f, .rel_height = 1.0f},
          .text     = "Host:",
          .center_x = false,
          .center_y = true});

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.6f, .rel_height = 1.0f},
          .text     = string,
          .center_x = false,
          .center_y = true});

      char text[256];

      double time;
      LUM_FAILURE_HANDLE(luminary_host_get_queue_time(host, &time));

      sprintf(text, "(%.2fs)", time);

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.2f, .rel_height = 1.0f},
          .text     = text,
          .center_x = false,
          .center_y = true});
    }
    window_pop_section(window);
  }

  LUM_FAILURE_HANDLE(luminary_host_get_device_queue_string(host, &string));

  if (string != (const char*) 0) {
    window_push_section(window, 24, 0);
    {
      window_margin(window, 8);

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.2f, .rel_height = 1.0f},
          .text     = "Device:",
          .center_x = false,
          .center_y = true});

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.6f, .rel_height = 1.0f},
          .text     = string,
          .center_x = false,
          .center_y = true});

      char text[256];

      double time;
      LUM_FAILURE_HANDLE(luminary_host_get_device_queue_time(host, &time));

      sprintf(text, "(%.2fs)", time);

      element_text(
        window, display,
        (ElementTextArgs){
          .color    = 0xFFFFFFFF,
          .size     = (ElementSize){.is_relative = true, .rel_width = 0.2f, .rel_height = 1.0f},
          .text     = text,
          .center_x = false,
          .center_y = true});
    }
    window_pop_section(window);
  }

  return false;
}

void window_renderer_status_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 1920 - 32 - 320;
  (*window)->y             = 1080 - 32 - 48;  // TODO: Implement fixed position windows with computed positions.
  (*window)->width         = 320;
  (*window)->height        = 48;
  (*window)->padding       = 0;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->background    = true;
  (*window)->action_func   = _window_renderer_status_action;

  window_allocate_memory(*window);
}
