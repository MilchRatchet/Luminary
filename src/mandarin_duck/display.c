#include "display.h"

static uint32_t __num_displays = 0;

static void _display_handle_resize(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_GetWindowSizeInPixels(display->sdl_window, (int*) &(display->width), (int*) &(display->height));

  SDL_Surface* surface = SDL_GetWindowSurface(display->sdl_window);

  display->buffer = surface->pixels;
  display->ld     = (uint32_t) surface->pitch;
}

static void _display_blit_to_display_buffer(Display* display, uint8_t* output) {
  for (uint32_t y = 0; y < display->height; y++) {
    memcpy(display->buffer + y * display->ld, output + y * sizeof(uint8_t) * 4 * display->width, sizeof(uint8_t) * 4 * display->width);
  }
}

void display_create(Display** _display, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(_display);

  if (__num_displays == 0) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
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

  rect.w = min(rect.w, (int) width);
  rect.h = min(rect.h, (int) height);

  info_message("Size: %d %d", rect.w, rect.h);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetStringProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "MandarinDuck - Using Luminary");
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, rect.w);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, rect.h);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_BORDERLESS_BOOLEAN, 1);

  display->sdl_window = SDL_CreateWindowWithProperties(sdl_properties);

  SDL_DestroyProperties(sdl_properties);

  if (!display->sdl_window) {
    crash_message("Failed to create SDL_Window.");
  }

  _display_handle_resize(display);

  keyboard_state_create(&display->keyboard_state);
  mouse_state_create(&display->mouse_state);
  camera_handler_create(&display->camera_handler);
  user_interface_create(&display->ui);

  *_display = display;
}

void display_query_events(Display* display, bool* exit_requested, bool* dirty) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(exit_requested);

  if (display->exit_requested) {
    *exit_requested = true;
    return;
  }

  keyboard_state_reset_phases(display->keyboard_state);
  mouse_state_reset_motion(display->mouse_state);
  mouse_state_reset_button(display->mouse_state);

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
        // TODO: Implement support for dragging and dropping files into Mandarin Duck.
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

void display_handle_inputs(Display* display, LuminaryHost* host, float time_step) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (display->keyboard_state->keys[SDL_SCANCODE_E].phase == KEY_PHASE_RELEASED) {
    display->show_ui = !display->show_ui;
    SDL_SetWindowRelativeMouseMode(display->sdl_window, !display->show_ui);
  }

  if (display->show_ui) {
    user_interface_handle_inputs(display->ui, display, host);
  }
  else {
    camera_handler_update(display->camera_handler, host, display->keyboard_state, display->mouse_state, time_step);
    _display_move_sun(display, host, time_step);
  }
}

static void _display_render_output(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputHandle output_handle;
  LUM_FAILURE_HANDLE(luminary_host_acquire_output(host, &output_handle));

  uint8_t* output_buffer;
  LUM_FAILURE_HANDLE(luminary_host_get_output_buffer(host, output_handle, (void**) &output_buffer));

  // No output buffer means that the renderer has never produced an output image.
  if (!output_buffer) {
    for (uint32_t y = 0; y < display->height; y++) {
      memset(display->buffer + y * display->ld, 0xFF, sizeof(uint8_t) * 4 * display->width);
    }

    LUM_FAILURE_HANDLE(luminary_host_release_output(host, output_handle));
    return;
  }

  _display_blit_to_display_buffer(display, output_buffer);

  LUM_FAILURE_HANDLE(luminary_host_release_output(host, output_handle));
}

void display_render(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  _display_render_output(display, host);

  if (display->show_ui) {
    user_interface_render(display->ui, display);
  }
}

void display_update(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_UpdateWindowSurface(display->sdl_window);
}

void display_destroy(Display** display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_DestroyWindow((*display)->sdl_window);

  keyboard_state_destroy(&(*display)->keyboard_state);
  mouse_state_destroy(&(*display)->mouse_state);
  camera_handler_destroy(&(*display)->camera_handler);
  user_interface_destroy(&(*display)->ui);

  LUM_FAILURE_HANDLE(host_free(display));

  __num_displays--;

  if (__num_displays == 0) {
    SDL_Quit();
  }
}
