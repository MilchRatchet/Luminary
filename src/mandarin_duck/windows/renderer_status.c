#include "renderer_status.h"

#include <stdio.h>
#include <string.h>

#include "display.h"
#include "elements/text.h"

static bool _window_renderer_status_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
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
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.2f, .height = 24},
          .text         = "Host:",
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = true});

      element_text(
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.6f, .height = 24},
          .text         = string,
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = false});

      char text[256];

      double time;
      LUM_FAILURE_HANDLE(luminary_host_get_queue_time(host, &time));

      sprintf(text, "(%.2fs)", time);

      element_text(
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.2f, .height = 24},
          .text         = text,
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = false});
    }
    window_pop_section(window);
  }

  LUM_FAILURE_HANDLE(luminary_host_get_device_queue_string(host, &string));

  if (string != (const char*) 0) {
    window_push_section(window, 24, 0);
    {
      window_margin(window, 8);

      element_text(
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.2f, .height = 24},
          .text         = "Device:",
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = true});

      element_text(
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.6f, .height = 24},
          .text         = string,
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = false});

      char text[256];

      double time;
      LUM_FAILURE_HANDLE(luminary_host_get_device_queue_time(host, &time));

      sprintf(text, "(%.2fs)", time);

      element_text(
        window, mouse_state,
        (ElementTextArgs){
          .color        = 0xFFFFFFFF,
          .size         = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.2f, .height = 24},
          .text         = text,
          .center_x     = false,
          .center_y     = true,
          .highlighting = false,
          .cache_text   = false});
    }
    window_pop_section(window);
  }

  return false;
}

void window_renderer_status_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 0;
  (*window)->y             = 0;
  (*window)->width         = 320;
  (*window)->height        = 48;
  (*window)->padding       = 0;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->is_movable    = false;
  (*window)->background    = true;
  (*window)->auto_size     = true;
  (*window)->auto_align    = true;
  (*window)->margins =
    (WindowMargins){.margin_bottom = 32, .margin_right = 32, .margin_left = WINDOW_MARGIN_INVALID, .margin_top = WINDOW_MARGIN_INVALID};
  (*window)->action_func = _window_renderer_status_action;
  (*window)->fixed_depth = true;

  window_allocate_memory(*window);
}
