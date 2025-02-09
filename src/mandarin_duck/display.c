#include "display.h"

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint32_t __num_displays = 0;

static void _display_handle_resize(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_GetWindowSizeInPixels(display->sdl_window, (int*) &(display->width), (int*) &(display->height));

  display->sdl_surface = SDL_GetWindowSurface(display->sdl_window);
  display->buffer      = display->sdl_surface->pixels;
  display->pitch       = (uint32_t) display->sdl_surface->pitch;
}

static void _display_blit_to_display_buffer(Display* display, LuminaryImage image) {
  uint8_t* buffer = image.buffer;

  const uint32_t width  = (display->width < image.width) ? display->width : image.width;
  const uint32_t height = (display->height < image.height) ? display->height : image.height;

  for (uint32_t y = 0; y < height; y++) {
    memcpy(display->buffer + y * display->pitch, buffer + y * image.ld * sizeof(LuminaryARGB8), sizeof(LuminaryARGB8) * width);
  }
}

static SDL_HitTestResult _display_sdl_hittestcallback(SDL_Window* window, const SDL_Point* area, void* data) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(area);

  Display* display = (Display*) data;

  const float prev_mouse_x = display->mouse_state->x;
  const float prev_mouse_y = display->mouse_state->y;

  display->mouse_state->x = area->x;
  display->mouse_state->y = area->y;

  bool mouse_hovers_background = false;
  user_interface_mouse_hovers_background(display->ui, display, &mouse_hovers_background);

  display->mouse_state->x = prev_mouse_x;
  display->mouse_state->y = prev_mouse_y;

  const bool alt_down = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;

  return (mouse_hovers_background && !alt_down) ? SDL_HITTEST_DRAGGABLE : SDL_HITTEST_NORMAL;
}

static void _display_set_hittest(Display* display, bool enable) {
  SDL_SetWindowHitTest(display->sdl_window, (SDL_HitTest) enable ? _display_sdl_hittestcallback : 0, (void*) display);
}

void display_create(Display** _display, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(_display);

  if (__num_displays == 0) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    TTF_Init();
  }

  __num_displays++;

  Display* display;
  LUM_FAILURE_HANDLE(host_malloc(&display, sizeof(Display)));
  memset(display, 0, sizeof(Display));

  display->show_ui = true;

  SDL_Rect rect;
  SDL_Rect screen_size;

  int display_count;
  SDL_DisplayID* displays = SDL_GetDisplays(&display_count);

  if (display_count == 0) {
    crash_message("No displays available.");
  }

  SDL_GetDisplayUsableBounds(displays[0], &rect);
  SDL_GetDisplayBounds(displays[0], &screen_size);

  // Add some margin to the usable bounds.
  const uint32_t margin = 5;

  rect.w = rect.w - 2 * margin;
  rect.h = rect.h - 2 * margin;

  rect.w = min(rect.w, (int) width);
  rect.h = min(rect.h, (int) height);

  // Make sure that the aspect ratio is maintained.
  const float aspect_ratio = ((float) width) / ((float) height);

  rect.w = rect.h * aspect_ratio;

  info_message("Size: %d %d", rect.w, rect.h);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetStringProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "MandarinDuck - Using Luminary");
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, rect.w);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, rect.h);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_BORDERLESS_BOOLEAN, true);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TRANSPARENT_BOOLEAN, true);

  display->sdl_window = SDL_CreateWindowWithProperties(sdl_properties);

  SDL_DestroyProperties(sdl_properties);

  if (!display->sdl_window) {
    crash_message("Failed to create SDL_Window.");
  }

  SDL_SetWindowSurfaceVSync(display->sdl_window, SDL_WINDOW_SURFACE_VSYNC_ADAPTIVE);

  _display_handle_resize(display);

  keyboard_state_create(&display->keyboard_state);
  mouse_state_create(&display->mouse_state);
  camera_handler_create(&display->camera_handler);
  user_interface_create(&display->ui);
  ui_renderer_create(&display->ui_renderer);
  text_renderer_create(&display->text_renderer);

  display->selected_cursor = SDL_SYSTEM_CURSOR_DEFAULT;
  for (uint32_t cursor_id = 0; cursor_id < SDL_SYSTEM_CURSOR_COUNT; cursor_id++) {
    display->sdl_cursors[cursor_id] = SDL_CreateSystemCursor(cursor_id);
  }

  display->mouse_mode = DISPLAY_MOUSE_MODE_DEFAULT;

  display_set_mouse_visible(display, display->show_ui);

  display->output_promise_handle = LUMINARY_OUTPUT_HANDLE_INVALID;

  *_display = display;
}

void display_set_mouse_visible(Display* display, bool enable) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->mouse_visible = enable;
  SDL_SetWindowRelativeMouseMode(display->sdl_window, !enable);
  _display_set_hittest(display, enable && (display->mouse_mode == DISPLAY_MOUSE_MODE_DEFAULT) && !display->active_camera_movement);
}

void display_set_cursor(Display* display, SDL_SystemCursor cursor) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->selected_cursor = cursor;
}

void display_set_mouse_mode(Display* display, DisplayMouseMode mouse_mode) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->mouse_mode = mouse_mode;

  // Invalidate the result
  if (mouse_mode == DISPLAY_MOUSE_MODE_FOCUS) {
    display->focus_pixel_data.pixel_query_is_valid = false;
  }

  _display_set_hittest(display, (mouse_mode == DISPLAY_MOUSE_MODE_DEFAULT));
}

static void _display_handle_drop_event(DisplayFileDrop** file_drop_array, SDL_DropEvent event) {
  if (event.type == SDL_EVENT_DROP_FILE) {
    DisplayFileDrop file_drop;
    file_drop.file_path = event.data;

    LUM_FAILURE_HANDLE(array_push(file_drop_array, &file_drop));
  }
}

void display_query_events(Display* display, DisplayFileDrop** file_drop_array, bool* exit_requested, bool* dirty) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(exit_requested);

  if (display->exit_requested) {
    *exit_requested = true;
    return;
  }

  keyboard_state_reset_phases(display->keyboard_state);
  mouse_state_reset_motion(display->mouse_state);
  mouse_state_step_phase(display->mouse_state);

  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    switch (event.type) {
      case SDL_EVENT_QUIT:
        *exit_requested = true;
        break;
      case SDL_EVENT_WINDOW_RESIZED:
        _display_handle_resize(display);
        *dirty = true;
        break;
      case SDL_EVENT_WINDOW_MOUSE_ENTER:
        break;
      case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        mouse_state_invalidate(display->mouse_state);
        break;
      case SDL_EVENT_WINDOW_HIT_TEST:
        break;
      case SDL_EVENT_KEY_DOWN:
      case SDL_EVENT_KEY_UP:
        keyboard_state_update(display->keyboard_state, event.key);
        break;
      case SDL_EVENT_MOUSE_MOTION:
        mouse_state_update_motion(display->mouse_state, event.motion);
        break;
      case SDL_EVENT_MOUSE_BUTTON_DOWN:
      case SDL_EVENT_MOUSE_BUTTON_UP:
        mouse_state_update_button(display->mouse_state, event.button);
        break;
      case SDL_EVENT_MOUSE_WHEEL:
        mouse_state_update_wheel(display->mouse_state, event.wheel);
        break;
      case SDL_EVENT_DROP_FILE:
        _display_handle_drop_event(file_drop_array, event.drop);
        break;
      default:
        warn_message("Unhandled SDL event type: %u.", event.type);
        break;
    }
  }
}

// TODO: Refactor
static void _display_move_sun(Display* display, LuminaryHost* host, float time_step) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminarySky sky;
  LUM_FAILURE_HANDLE(luminary_host_get_sky(host, &sky));

  if (display->keyboard_state->keys[SDL_SCANCODE_LEFT].down) {
    sky.azimuth -= 0.2f * time_step;
  }

  if (display->keyboard_state->keys[SDL_SCANCODE_RIGHT].down) {
    sky.azimuth += 0.2f * time_step;
  }

  if (display->keyboard_state->keys[SDL_SCANCODE_UP].down) {
    sky.altitude += 0.2f * time_step;
  }

  if (display->keyboard_state->keys[SDL_SCANCODE_DOWN].down) {
    sky.altitude -= 0.2f * time_step;
  }

  LUM_FAILURE_HANDLE(luminary_host_set_sky(host, &sky));
}

static void _display_set_default_cursor(Display* display) {
  SDL_SystemCursor cursor;

  bool mouse_hovers_background = false;
  user_interface_mouse_hovers_background(display->ui, display, &mouse_hovers_background);

  switch (display->mouse_mode) {
    case DISPLAY_MOUSE_MODE_DEFAULT:
    default:
      cursor = SDL_SYSTEM_CURSOR_DEFAULT;
      break;
    case DISPLAY_MOUSE_MODE_SELECT:
    case DISPLAY_MOUSE_MODE_FOCUS:
      cursor = (mouse_hovers_background) ? SDL_SYSTEM_CURSOR_POINTER : SDL_SYSTEM_CURSOR_DEFAULT;
      break;
  }

  display_set_cursor(display, cursor);
}

static void _display_query_pixel_info(Display* display, LuminaryHost* host, float request_x, float request_y) {
  MD_CHECK_NULL_ARGUMENT(display);

  const float rel_x = request_x / display->width;
  const float rel_y = request_y / display->height;

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  const uint16_t x = rel_x * settings.width;
  const uint16_t y = rel_y * settings.height;

  LuminaryPixelQueryResult result;

  LUM_FAILURE_HANDLE(luminary_host_get_pixel_info(host, x, y, &result));

  display->reference_x = request_x;
  display->reference_y = request_y;

  display->awaiting_pixel_query_result = !result.pixel_query_is_valid;

  switch (display->mouse_mode) {
    case DISPLAY_MOUSE_MODE_DEFAULT:
      display->move_pixel_data = result;
      break;
    case DISPLAY_MOUSE_MODE_SELECT:
      display->select_pixel_data = result;
      break;
    case DISPLAY_MOUSE_MODE_FOCUS:
      display->focus_pixel_data = result;
      break;
    default:
      break;
  }
}

static void _display_start_camera_mode(Display* display, const bool left_pressed) {
  display->active_camera_movement = true;
  display_set_mouse_visible(display, false);

  camera_handler_set_mode(display->camera_handler, (left_pressed) ? CAMERA_MODE_ORBIT : CAMERA_MODE_ZOOM);
  camera_handler_set_reference_pos(display->camera_handler, display->move_pixel_data.rel_hit_pos);
}

static WindowVisibilityMask _display_get_window_visibility_mask(Display* display) {
  if (display->show_ui) {
    return (display->active_camera_movement) ? WINDOW_VISIBILITY_MASK_MOVEMENT : WINDOW_VISIBILITY_MASK_ALL;
  }

  return WINDOW_VISIBILITY_NONE;
}

static void _display_generate_screenshot_name(const char* output_directory, char* string, LuminaryImage image) {
  time_t rawtime;
  struct tm timeinfo;

  char time_string[2048];

  time(&rawtime);
  timeinfo = *localtime(&rawtime);
  strftime(time_string, 2048, "%Y-%m-%d-%H-%M-%S", &timeinfo);

  sprintf(string, "%s/Snap-%s-%05u-%07.1fs.png", output_directory, time_string, image.meta_data.sample_count, image.meta_data.time);
}

static void _display_generate_screenshot(Display* display, LuminaryHost* host) {
  if (display->output_promise_handle != LUMINARY_OUTPUT_HANDLE_INVALID)
    return;

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  LuminaryOutputRequestProperties properties;
  properties.sample_count = 0;
  properties.width        = settings.width;
  properties.height       = settings.height;

  LUM_FAILURE_HANDLE(luminary_host_request_output(host, properties, &display->output_promise_handle));
}

void display_handle_inputs(Display* display, LuminaryHost* host, float time_step) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  display->frametime = time_step;

  if (display->keyboard_state->keys[SDL_SCANCODE_E].phase == KEY_PHASE_RELEASED) {
    display->show_ui                = !display->show_ui;
    display->active_camera_movement = !display->show_ui;
    display_set_mouse_visible(display, display->show_ui);
  }

  if (display->keyboard_state->keys[SDL_SCANCODE_F2].phase == KEY_PHASE_PRESSED) {
    _display_generate_screenshot(display, host);
  }

  if (display->awaiting_pixel_query_result) {
    display->awaiting_pixel_query_result = false;

    switch (display->mouse_mode) {
      case DISPLAY_MOUSE_MODE_DEFAULT:
      default: {
        const bool left_down  = display->mouse_state->down;
        const bool right_down = display->mouse_state->right_down;
        const bool alt_down   = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;

        if ((left_down || right_down) && alt_down) {
          _display_query_pixel_info(display, host, display->reference_x, display->reference_y);

          if (display->move_pixel_data.pixel_query_is_valid) {
            _display_start_camera_mode(display, left_down);
          }
        }
      } break;
      case DISPLAY_MOUSE_MODE_SELECT:
      case DISPLAY_MOUSE_MODE_FOCUS:
        _display_query_pixel_info(display, host, display->reference_x, display->reference_y);
        break;
    }
  }

  _display_move_sun(display, host, time_step);

  _display_set_default_cursor(display);

  WindowVisibilityMask visibility_mask = _display_get_window_visibility_mask(display);
  const bool ui_handled_mouse          = user_interface_handle_inputs(display->ui, display, host, visibility_mask);

  if (display->show_ui) {
    if (display->keyboard_state->keys[SDL_SCANCODE_1].down) {
      display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_DEFAULT);
    }

    if (display->keyboard_state->keys[SDL_SCANCODE_2].down) {
      display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_SELECT);
    }

    if (display->keyboard_state->keys[SDL_SCANCODE_3].down) {
      display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_FOCUS);
    }

    if (!ui_handled_mouse) {
      switch (display->mouse_mode) {
        case DISPLAY_MOUSE_MODE_DEFAULT:
        default:
          if (display->mouse_visible) {
            const bool left_pressed  = display->mouse_state->phase == MOUSE_PHASE_PRESSED;
            const bool right_pressed = display->mouse_state->right_phase == MOUSE_PHASE_PRESSED;
            const bool alt_down      = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;

            if ((left_pressed || right_pressed) && alt_down) {
              _display_query_pixel_info(display, host, display->mouse_state->x, display->mouse_state->y);

              if (display->move_pixel_data.pixel_query_is_valid) {
                _display_start_camera_mode(display, left_pressed);
              }
            }
          }
          else if (display->active_camera_movement) {
            const bool left_pressed  = display->mouse_state->down;
            const bool right_pressed = display->mouse_state->right_down;
            const bool alt_down      = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;

            const bool cond0 = (display->camera_handler->mode == CAMERA_MODE_ORBIT && !left_pressed);
            const bool cond1 = (display->camera_handler->mode == CAMERA_MODE_ZOOM && !right_pressed);

            if (cond0 || cond1 || !alt_down) {
              display->active_camera_movement = !display->show_ui;
              display_set_mouse_visible(display, true);

              camera_handler_set_mode(display->camera_handler, CAMERA_MODE_FLY);
            }
          }
          break;
        case DISPLAY_MOUSE_MODE_SELECT:
        case DISPLAY_MOUSE_MODE_FOCUS:
          if (display->mouse_state->phase == MOUSE_PHASE_PRESSED) {
            _display_query_pixel_info(display, host, display->mouse_state->x, display->mouse_state->y);
          }
          break;
      }
    }

    if (display->mouse_mode == DISPLAY_MOUSE_MODE_FOCUS && display->focus_pixel_data.pixel_query_is_valid) {
      if (display->focus_pixel_data.depth != FLT_MAX && display->focus_pixel_data.depth > 0) {
        LuminaryCamera camera;
        LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

        camera.focal_length = display->focus_pixel_data.depth;

        LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));

        // Invalidate the pixel query so that it does not keep overriding the focal length now.
        display->focus_pixel_data.pixel_query_is_valid = false;
      }
    }
  }

  if (display->active_camera_movement) {
    camera_handler_update(display->camera_handler, host, display->keyboard_state, display->mouse_state, time_step);
  }
}

void display_handle_outputs(Display* display, LuminaryHost* host, const char* output_directory) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (display->output_promise_handle != LUMINARY_OUTPUT_HANDLE_INVALID) {
    LuminaryOutputHandle output_handle;
    luminary_host_try_await_output(host, display->output_promise_handle, &output_handle);

    if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
      return;

    LuminaryImage output_image;
    LUM_FAILURE_HANDLE(luminary_host_get_image(host, output_handle, &output_image));

    LuminaryPath* image_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&image_path));

    char string[4096];
    _display_generate_screenshot_name(output_directory, string, output_image);

    LUM_FAILURE_HANDLE(luminary_path_set_from_string(image_path, string));

    LUM_FAILURE_HANDLE(luminary_host_save_png(host, output_image, image_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&image_path));

    LUM_FAILURE_HANDLE(luminary_host_release_output(host, output_handle));

    display->output_promise_handle = LUMINARY_OUTPUT_HANDLE_INVALID;
  }
}

static void _display_render_output(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputHandle output_handle;
  LUM_FAILURE_HANDLE(luminary_host_acquire_output(host, &output_handle));

  if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID) {
    for (uint32_t y = 0; y < display->height; y++) {
      memset(display->buffer + y * display->pitch, 0xFF, sizeof(uint8_t) * 4 * display->width);
    }

    return;
  }

  LuminaryImage output_image;
  LUM_FAILURE_HANDLE(luminary_host_get_image(host, output_handle, &output_image));

  display->current_render_meta_data.elapsed_time = output_image.meta_data.time;
  display->current_render_meta_data.sample_count = output_image.meta_data.sample_count;

  _display_blit_to_display_buffer(display, output_image);

  LUM_FAILURE_HANDLE(luminary_host_release_output(host, output_handle));
}

void display_render(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  _display_render_output(display, host);

  WindowVisibilityMask visibility_mask = _display_get_window_visibility_mask(display);
  user_interface_render(display->ui, display, visibility_mask);
}

void display_update(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_SetCursor(display->sdl_cursors[display->selected_cursor]);

  SDL_UpdateWindowSurface(display->sdl_window);
}

void display_destroy(Display** display) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(*display);

  SDL_SetCursor((SDL_Cursor*) 0);

  for (uint32_t cursor_id = 0; cursor_id < SDL_SYSTEM_CURSOR_COUNT; cursor_id++) {
    SDL_DestroyCursor((*display)->sdl_cursors[cursor_id]);
  }

  SDL_DestroyWindow((*display)->sdl_window);

  keyboard_state_destroy(&(*display)->keyboard_state);
  mouse_state_destroy(&(*display)->mouse_state);
  camera_handler_destroy(&(*display)->camera_handler);
  user_interface_destroy(&(*display)->ui);
  ui_renderer_destroy(&(*display)->ui_renderer);
  text_renderer_destroy(&(*display)->text_renderer);

  LUM_FAILURE_HANDLE(host_free(display));

  __num_displays--;

  if (__num_displays == 0) {
    SDL_Quit();
    TTF_Quit();
  }
}
