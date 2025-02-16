#ifndef MANDARIN_DUCK_DISPLAY_H
#define MANDARIN_DUCK_DISPLAY_H

#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

#include "camera_handler.h"
#include "keyboard_state.h"
#include "mouse_state.h"
#include "render_region.h"
#include "text_renderer.h"
#include "ui_renderer.h"
#include "user_interface.h"
#include "utils.h"

enum DisplayMouseMode {
  DISPLAY_MOUSE_MODE_DEFAULT,
  DISPLAY_MOUSE_MODE_SELECT,
  DISPLAY_MOUSE_MODE_FOCUS,
  DISPLAY_MOUSE_MODE_RENDER_REGION,
  DISPLAY_MOUSE_MODE_COUNT
} typedef DisplayMouseMode;

struct RenderMetaData {
  float elapsed_time;
  uint32_t sample_count;
} typedef RenderMetaData;

struct DisplayFileDrop {
  const char* file_path;
} typedef DisplayFileDrop;

struct Display {
  SDL_Window* sdl_window;
  SDL_Surface* sdl_surface;
  uint32_t width;
  uint32_t height;
  uint8_t* buffer;
  uint32_t pitch;
  bool exit_requested;
  bool show_ui;
  bool mouse_visible;
  bool active_camera_movement;
  double frametime;
  KeyboardState* keyboard_state;
  MouseState* mouse_state;
  CameraHandler* camera_handler;
  UserInterface* ui;
  UIRenderer* ui_renderer;
  TextRenderer* text_renderer;
  SDL_Cursor* sdl_cursors[SDL_SYSTEM_CURSOR_COUNT];
  SDL_SystemCursor selected_cursor;
  DisplayMouseMode mouse_mode;
  LuminaryPixelQueryResult move_pixel_data;
  LuminaryPixelQueryResult select_pixel_data;
  LuminaryPixelQueryResult focus_pixel_data;
  bool awaiting_pixel_query_result;
  uint32_t reference_x;
  uint32_t reference_y;
  RenderMetaData current_render_meta_data;
  LuminaryOutputPromiseHandle output_promise_handle;
  RenderRegion* region;
} typedef Display;

void display_create(Display** _display, uint32_t width, uint32_t height);
void display_set_mouse_visible(Display* display, bool enable);
void display_set_cursor(Display* display, SDL_SystemCursor cursor);
void display_set_mouse_mode(Display* display, DisplayMouseMode mouse_mode);
void display_query_events(Display* display, DisplayFileDrop** file_drop_array, bool* exit_requested, bool* dirty);
void display_handle_inputs(Display* display, LuminaryHost* host, float time_step);
void display_handle_outputs(Display* display, LuminaryHost* host, const char* output_directory);
void display_render(Display* display, LuminaryHost* host);
void display_update(Display* display);
void display_destroy(Display** display);

#endif /* MANDARIN_DUCK_DISPLAY_H */
