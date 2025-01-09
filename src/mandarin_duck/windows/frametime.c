#include "frametime.h"

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "display.h"
#include "elements/text.h"

static bool _window_frametime_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  window_push_section(window, 24, 0);
  {
    window_margin(window, 8);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.35f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = "Luminary:",
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = false});

    char text[256];

    double time;
    LUM_FAILURE_HANDLE(luminary_host_get_current_sample_time(host, &time));

    sprintf(text, "%.1f SPPPS", 1.0 / time);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.325f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false});

    sprintf(text, "(%.2fms)", 1000.0 * time);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.325f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false});
  }
  window_pop_section(window);

  window_push_section(window, 24, 0);
  {
    window_margin(window, 8);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.35f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = "Mandarin Duck:",
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = true,
        .auto_size    = false,
        .is_clickable = false});

    char text[256];

    sprintf(text, "%.1f FPS", 1.0 / display->frametime);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.325f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false});

    sprintf(text, "(%.2fms)", 1000.0 * display->frametime);

    element_text(
      window, display, mouse_state,
      (ElementTextArgs){
        .color    = 0xFFFFFFFF,
        .size     = (ElementSize){.width = ELEMENT_SIZE_INVALID, .rel_width = 0.325f, .height = ELEMENT_SIZE_INVALID, .rel_height = 1.0f},
        .text     = text,
        .center_x = false,
        .center_y = true,
        .highlighting = false,
        .cache_text   = false,
        .auto_size    = false,
        .is_clickable = false});
  }
  window_pop_section(window);

  return false;
}

void window_frametime_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type            = WINDOW_TYPE_FRAMETIME;
  (*window)->visibility_mask = WINDOW_VISIBILITY_STATUS;
  (*window)->x               = 0;
  (*window)->y               = 0;
  (*window)->width           = 320;
  (*window)->height          = 48;
  (*window)->padding         = 0;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = true;
  (*window)->is_movable      = false;
  (*window)->background      = true;
  (*window)->auto_size       = false;
  (*window)->auto_align      = true;
  (*window)->margins =
    (WindowMargins){.margin_bottom = 32, .margin_left = 32, .margin_right = WINDOW_MARGIN_INVALID, .margin_top = WINDOW_MARGIN_INVALID};
  (*window)->action_func = _window_frametime_action;
  (*window)->fixed_depth = true;

  window_allocate_memory(*window);
}
