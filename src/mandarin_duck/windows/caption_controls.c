#include "caption_controls.h"

#include "display.h"
#include "elements/button.h"

#define CAPTION_CONTROL_BUTTON_SIZE 16

static void _window_caption_controls_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (element_button(
        window, display, mouse_state,
        (ElementButtonArgs) {.size         = (ElementSize) {.width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
                             .shape        = ELEMENT_BUTTON_SHAPE_CIRCLE,
                             .color        = 0xFFFF5F57,
                             .hover_color  = 0xFFFF1411,
                             .press_color  = 0xFF720000,
                             .tooltip_text = (const char*) 0})) {
    display->exit_requested = true;
  }

  window_margin(window, CAPTION_CONTROL_BUTTON_SIZE);

  if (element_button(
        window, display, mouse_state,
        (ElementButtonArgs) {.size         = (ElementSize) {.width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
                             .shape        = ELEMENT_BUTTON_SHAPE_CIRCLE,
                             .color        = 0xFFFFBB2F,
                             .hover_color  = 0xFFFFD51A,
                             .press_color  = 0xFF975600,
                             .tooltip_text = (const char*) 0})) {
    SDL_MinimizeWindow(display->sdl_window);
  }

  window_margin(window, CAPTION_CONTROL_BUTTON_SIZE);

  if (element_button(
        window, display, mouse_state,
        (ElementButtonArgs) {.size         = (ElementSize) {.width = CAPTION_CONTROL_BUTTON_SIZE, .height = CAPTION_CONTROL_BUTTON_SIZE},
                             .shape        = ELEMENT_BUTTON_SHAPE_CIRCLE,
                             .color        = 0xFF26C942,
                             .hover_color  = 0xFF13FF21,
                             .press_color  = 0xFF026200,
                             .tooltip_text = (const char*) 0})) {
    uint32_t width;
    uint32_t height;
    uint32_t x;
    uint32_t y;

    if (display->is_maximized) {
      width  = display->unmaximize_width;
      height = display->unmaximize_height;

      x = display->unmaximize_x;
      y = display->unmaximize_y;
    }
    else {
      // Store the current size so we can restore it later again.
      display->unmaximize_width  = display->width;
      display->unmaximize_height = display->height;

      width  = display->screen_width;
      height = display->screen_height;

      SDL_GetWindowPosition(display->sdl_window, (int*) &display->unmaximize_x, (int*) &display->unmaximize_y);

      x = SDL_WINDOWPOS_CENTERED;
      y = SDL_WINDOWPOS_CENTERED;
    }

    display->is_maximized = !display->is_maximized;

    SDL_SetWindowSize(display->sdl_window, (int) width, (int) height);
    SDL_SetWindowPosition(display->sdl_window, (int) x, (int) y);

    if (display->sync_render_resolution) {
      LuminaryRendererSettings settings;
      LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

      if (settings.width != width || settings.height != height) {
        settings.width  = width;
        settings.height = height;
        LUM_FAILURE_HANDLE(luminary_host_set_settings(host, &settings));
      }
    }
  }
}

void window_caption_controls_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type            = WINDOW_TYPE_CAPTION_CONTROLS;
  (*window)->visibility_mask = WINDOW_VISIBILITY_CAPTION_CONTROLS;
  (*window)->x               = CAPTION_CONTROL_BUTTON_SIZE * 2;
  (*window)->y               = CAPTION_CONTROL_BUTTON_SIZE * 2;
  (*window)->width           = CAPTION_CONTROL_BUTTON_SIZE * 5;
  (*window)->height          = CAPTION_CONTROL_BUTTON_SIZE;
  (*window)->padding         = 0;
  (*window)->is_horizontal   = true;
  (*window)->is_visible      = true;
  (*window)->is_movable      = false;
  (*window)->background      = false;
  (*window)->auto_size       = false;
  (*window)->auto_align      = true;
  (*window)->margins =
    (WindowMargins) {.margin_top = 32, .margin_left = 32, .margin_right = WINDOW_MARGIN_INVALID, .margin_bottom = WINDOW_MARGIN_INVALID};
  (*window)->action_func = _window_caption_controls_action;
  (*window)->fixed_depth = true;

  window_allocate_memory(*window);
}
