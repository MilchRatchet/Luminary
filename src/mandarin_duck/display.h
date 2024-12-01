#ifndef MANDARIN_DUCK_DISPLAY_H
#define MANDARIN_DUCK_DISPLAY_H

#include <SDL3/SDL.h>

#include "camera_handler.h"
#include "keyboard_state.h"
#include "mouse_state.h"
#include "utils.h"

struct Display {
  SDL_Window* sdl_window;
  uint32_t width;
  uint32_t height;
  uint8_t* buffer;
  uint32_t ld;
  bool show_ui;
  KeyboardState* keyboard_state;
  MouseState* mouse_state;
  CameraHandler* camera_handler;
} typedef Display;

void display_create(Display** _display, uint32_t width, uint32_t height);
void display_query_events(Display* display, bool* exit_requested, bool* dirty);
void display_handle_inputs(Display* display, LuminaryHost* host);
void display_render(Display* display, LuminaryHost* host);
void display_update(Display* display);
void display_destroy(Display** display);

#endif /* MANDARIN_DUCK_DISPLAY_H */
