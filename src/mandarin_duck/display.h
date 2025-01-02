#ifndef MANDARIN_DUCK_DISPLAY_H
#define MANDARIN_DUCK_DISPLAY_H

#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

#include "camera_handler.h"
#include "keyboard_state.h"
#include "mouse_state.h"
#include "text_renderer.h"
#include "ui_renderer.h"
#include "user_interface.h"
#include "utils.h"

struct Display {
  SDL_Window* sdl_window;
  SDL_Surface* sdl_surface;
  uint32_t width;
  uint32_t height;
  uint8_t* buffer;
  uint32_t ld;
  bool exit_requested;
  bool show_ui;
  double frametime;
  KeyboardState* keyboard_state;
  MouseState* mouse_state;
  CameraHandler* camera_handler;
  UserInterface* ui;
  UIRenderer* ui_renderer;
  TextRenderer* text_renderer;
  SDL_Cursor* sdl_cursors[SDL_SYSTEM_CURSOR_COUNT];
  SDL_SystemCursor selected_cursor;
} typedef Display;

void display_create(Display** _display, uint32_t width, uint32_t height);
void display_set_mouse_visible(Display* display, bool enable);
void display_set_cursor(Display* display, SDL_SystemCursor cursor);
void display_query_events(Display* display, bool* exit_requested, bool* dirty);
void display_handle_inputs(Display* display, LuminaryHost* host, float time_step);
void display_render(Display* display, LuminaryHost* host);
void display_update(Display* display);
void display_destroy(Display** display);

#endif /* MANDARIN_DUCK_DISPLAY_H */
