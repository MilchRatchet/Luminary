#include "caption_controls.h"

#include "display.h"
#include "elements/button.h"

#define CAPTION_CONTROL_BUTTON_SIZE 32

static void _window_caption_controls_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .size  = (ElementSize){.is_relative = false, .width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
          .shape = ELEMENT_BUTTON_SHAPE_CIRCLE,
          .color = 0xFFFF0000})) {
    display->exit_requested = true;
  }

  window_margin(window, CAPTION_CONTROL_BUTTON_SIZE);

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .size  = (ElementSize){.is_relative = false, .width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
          .shape = ELEMENT_BUTTON_SHAPE_CIRCLE,
          .color = 0xFFFF0000})) {
    SDL_MinimizeWindow(display->sdl_window);
  }
}

void window_caption_controls_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = CAPTION_CONTROL_BUTTON_SIZE;
  (*window)->y             = CAPTION_CONTROL_BUTTON_SIZE;
  (*window)->width         = CAPTION_CONTROL_BUTTON_SIZE * 5;
  (*window)->height        = CAPTION_CONTROL_BUTTON_SIZE;
  (*window)->padding       = 0;
  (*window)->is_horizontal = true;
  (*window)->is_visible    = true;
  (*window)->background    = false;
  (*window)->action_func   = _window_caption_controls_action;
}
