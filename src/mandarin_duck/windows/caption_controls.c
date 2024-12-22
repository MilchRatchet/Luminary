#include "caption_controls.h"

#include "display.h"
#include "elements/button.h"

#define CAPTION_CONTROL_BUTTON_SIZE 16

static bool _window_caption_controls_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .size        = (ElementSize){.is_relative = false, .width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
          .shape       = ELEMENT_BUTTON_SHAPE_CIRCLE,
          .color       = 0xFFFF5F57,
          .hover_color = 0xFFFF1411,
          .press_color = 0xFF720000})) {
    display->exit_requested = true;
  }

  window_margin(window, CAPTION_CONTROL_BUTTON_SIZE);

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .size        = (ElementSize){.is_relative = false, .width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
          .shape       = ELEMENT_BUTTON_SHAPE_CIRCLE,
          .color       = 0xFFFFBB2F,
          .hover_color = 0xFFFFD51A,
          .press_color = 0xFF975600})) {
    SDL_MinimizeWindow(display->sdl_window);
  }

  window_margin(window, CAPTION_CONTROL_BUTTON_SIZE);

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .size        = (ElementSize){.is_relative = false, .width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
          .shape       = ELEMENT_BUTTON_SHAPE_CIRCLE,
          .color       = 0xFF26C942,
          .hover_color = 0xFF13FF21,
          .press_color = 0xFF026200})) {
  }

  return false;
}

void window_caption_controls_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = CAPTION_CONTROL_BUTTON_SIZE * 2;
  (*window)->y             = CAPTION_CONTROL_BUTTON_SIZE * 2;
  (*window)->width         = CAPTION_CONTROL_BUTTON_SIZE * 5;
  (*window)->height        = CAPTION_CONTROL_BUTTON_SIZE;
  (*window)->padding       = 0;
  (*window)->is_horizontal = true;
  (*window)->is_visible    = true;
  (*window)->background    = false;
  (*window)->action_func   = _window_caption_controls_action;

  window_allocate_memory(*window);
}
