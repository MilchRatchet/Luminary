#include <limits.h>
#include <string.h>

#include "display_internal.h"

#define DISPLAY_SCREEN_MARGIN 10

static uint32_t __num_displays = 0;

static void _display_get_initial_window_size(uint32_t requested_width, uint32_t requested_height, uint32_t* width, uint32_t* height) {
  *width  = (requested_width > 0) ? requested_width : 1920;
  *height = (requested_height > 0) ? requested_height : 1080;

  const SDL_DisplayID primary_display = SDL_GetPrimaryDisplay();
  SDL_Rect usable_bounds;

  if (primary_display == 0 || SDL_GetDisplayUsableBounds(primary_display, &usable_bounds) == false)
    return;

  const int32_t available_width  = (usable_bounds.w > 2 * DISPLAY_SCREEN_MARGIN) ? usable_bounds.w - 2 * DISPLAY_SCREEN_MARGIN : 1;
  const int32_t available_height = (usable_bounds.h > 2 * DISPLAY_SCREEN_MARGIN) ? usable_bounds.h - 2 * DISPLAY_SCREEN_MARGIN : 1;

  const float scale_factor = fminf(1.0f, fminf((float) available_width / *width, (float) available_height / *height));

  *width  = (uint32_t) (*width * scale_factor);
  *height = (uint32_t) (*height * scale_factor);

  if (*width == 0)
    *width = 1;
  if (*height == 0)
    *height = 1;
}

static void _display_refresh_surface(Display* display) {
  display->sdl_surface = SDL_GetWindowSurface(display->sdl_window);
  if (display->sdl_surface == NULL) {
    display->buffer = NULL;
    warn_message("Unable to acquire window surface: %s", SDL_GetError());
    return;
  }

  display->width  = (uint32_t) display->sdl_surface->w;
  display->height = (uint32_t) display->sdl_surface->h;
  display->buffer = display->sdl_surface->pixels;
  display->pitch  = (uint32_t) display->sdl_surface->pitch;

  display_zoom_handler_set_display_size(display->zoom_handler, display->width, display->height);
}

static void _display_clamp_window_position(Display* display) {
  const SDL_Rect bounds = display->screen_bounds;
  int window_x, window_y, width, height;
  if (
    bounds.w <= 0 || bounds.h <= 0 || SDL_GetWindowSize(display->sdl_window, &width, &height) == false
    || SDL_GetWindowPosition(display->sdl_window, &window_x, &window_y) == false)
    return;

  const int clamped_x = max(bounds.x, min(window_x, bounds.x + bounds.w - width));
  const int clamped_y = max(bounds.y, min(window_y, bounds.y + bounds.h - height));
  if (clamped_x != window_x || clamped_y != window_y)
    SDL_SetWindowPosition(display->sdl_window, clamped_x, clamped_y);
}

static void _display_handle_display_change(Display* display) {
  const SDL_DisplayID display_id = SDL_GetDisplayForWindow(display->sdl_window);
  SDL_Rect bounds;
  if (display_id == 0 || SDL_GetDisplayUsableBounds(display_id, &bounds) == false) {
    display->screen_bounds = (SDL_Rect) {0};
    warn_message("Unable to get usable window display bounds: %s", SDL_GetError());
    return;
  }

  if (bounds.w > 2 * DISPLAY_SCREEN_MARGIN) {
    bounds.x += DISPLAY_SCREEN_MARGIN;
    bounds.w -= 2 * DISPLAY_SCREEN_MARGIN;
  }
  if (bounds.h > 2 * DISPLAY_SCREEN_MARGIN) {
    bounds.y += DISPLAY_SCREEN_MARGIN;
    bounds.h -= 2 * DISPLAY_SCREEN_MARGIN;
  }
  display->screen_bounds = bounds;

  int width, height;
  if (SDL_GetWindowSize(display->sdl_window, &width, &height) && (width > bounds.w || height > bounds.h))
    display_resize(display, (uint32_t) width, (uint32_t) height);

  _display_clamp_window_position(display);
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

  if (mouse_hovers_background == false) {
    display->mouse_state->x = prev_mouse_x;
    display->mouse_state->y = prev_mouse_y;
  }

  const bool alt_down         = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;
  const bool ctrl_down        = display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down;
  const bool right_mouse_down = display->mouse_state->right_down;

  bool is_draggable = true;
  is_draggable &= mouse_hovers_background;
  is_draggable &= alt_down == false;
  is_draggable &= ctrl_down == false;
  is_draggable &= right_mouse_down == false;
  is_draggable &= display->is_maximized == false;

  return (is_draggable) ? SDL_HITTEST_DRAGGABLE : SDL_HITTEST_NORMAL;
}

static void _display_set_hittest(Display* display, bool enable) {
  SDL_SetWindowHitTest(display->sdl_window, (SDL_HitTest) enable ? _display_sdl_hittestcallback : 0, (void*) display);
}

void display_create(Display** _display, uint32_t requested_window_width, uint32_t requested_window_height) {
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

  uint32_t initial_window_width;
  uint32_t initial_window_height;
  _display_get_initial_window_size(requested_window_width, requested_window_height, &initial_window_width, &initial_window_height);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetStringProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "MandarinDuck - Using Luminary");
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, initial_window_width);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, initial_window_height);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_BORDERLESS_BOOLEAN, true);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TRANSPARENT_BOOLEAN, false);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);

  display->sdl_window = SDL_CreateWindowWithProperties(sdl_properties);

  SDL_DestroyProperties(sdl_properties);

  if (display->sdl_window == NULL) {
    crash_message("Failed to create SDL_Window.");
  }

  keyboard_state_create(&display->keyboard_state);
  mouse_state_create(&display->mouse_state);
  camera_handler_create(&display->camera_handler);
  user_interface_create(&display->ui);
  ui_renderer_create(&display->ui_renderer);
  text_renderer_create(&display->text_renderer);
  render_region_create(&display->region);
  display_zoom_handler_create(&display->zoom_handler);
  file_dialog_handler_create(&display->scene_file_path);
  file_dialog_handler_create(&display->entity_properties_path);

  _display_handle_display_change(display);
  _display_refresh_surface(display);
  if (display->sdl_surface != NULL)
    SDL_SetWindowSurfaceVSync(display->sdl_window, SDL_WINDOW_SURFACE_VSYNC_ADAPTIVE);

  display->selected_cursor = SDL_SYSTEM_CURSOR_DEFAULT;
  for (uint32_t cursor_id = 0; cursor_id < SDL_SYSTEM_CURSOR_COUNT; cursor_id++) {
    display->sdl_cursors[cursor_id] = SDL_CreateSystemCursor(cursor_id);
  }

  display->mouse_mode = DISPLAY_MOUSE_MODE_DEFAULT;

  display_set_mouse_visible(display, display->show_ui);

  display_render_init(display);

  LUM_FAILURE_HANDLE(array_create(&display->status_messages, sizeof(DisplayStatusMessage), 4));
  display_screenshot_init(display);

  *_display = display;
}

static bool _display_get_mouse_mode_enables_hittest(DisplayMouseMode mode) {
  return (mode == DISPLAY_MOUSE_MODE_DEFAULT) || (mode == DISPLAY_MOUSE_MODE_LOCK);
}

void display_set_mouse_visible(Display* display, bool enable) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->mouse_visible = enable;
  SDL_SetWindowRelativeMouseMode(display->sdl_window, enable == false);
  _display_set_hittest(
    display, enable && _display_get_mouse_mode_enables_hittest(display->mouse_mode) && display->active_camera_movement == false);
}

void display_set_mouse_mode(Display* display, DisplayMouseMode mouse_mode) {
  MD_CHECK_NULL_ARGUMENT(display);

  if (display->mouse_mode == mouse_mode) {
    if (mouse_mode != DISPLAY_MOUSE_MODE_DEFAULT)
      display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_DEFAULT);

    return;
  }

  display->mouse_mode = mouse_mode;

  // Invalidate the result
  if (mouse_mode == DISPLAY_MOUSE_MODE_FOCUS) {
    display->focus_pixel_data.pixel_query_is_valid = false;
  }

  _display_set_hittest(display, _display_get_mouse_mode_enables_hittest(display->mouse_mode));
}

static void _display_handle_drop_event(DisplayFileDrop** file_drop_array, SDL_DropEvent event) {
  if (event.type == SDL_EVENT_DROP_FILE) {
    DisplayFileDrop file_drop;
    file_drop.file_path = event.data;

    LUM_FAILURE_HANDLE(array_push(file_drop_array, &file_drop));
  }
}

void display_query_events(Display* display, DisplayFileDrop** file_drop_array, bool* exit_requested) {
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
      case SDL_EVENT_WINDOW_SHOWN:
      case SDL_EVENT_WINDOW_HIDDEN:
      case SDL_EVENT_WINDOW_MOVED:
        break;
      case SDL_EVENT_WINDOW_EXPOSED:
      case SDL_EVENT_WINDOW_RESTORED:
        if (display->sdl_surface == NULL)
          _display_refresh_surface(display);
        break;
      case SDL_EVENT_WINDOW_RESIZED:
        _display_clamp_window_position(display);
        _display_refresh_surface(display);
        break;
      case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        _display_refresh_surface(display);
        break;
      case SDL_EVENT_WINDOW_MINIMIZED:
      case SDL_EVENT_WINDOW_MAXIMIZED:
        break;
      case SDL_EVENT_WINDOW_MOUSE_ENTER:
        break;
      case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        mouse_state_invalidate(display->mouse_state);
        break;
      case SDL_EVENT_WINDOW_FOCUS_GAINED:
      case SDL_EVENT_WINDOW_FOCUS_LOST:
        break;
      case SDL_EVENT_WINDOW_HIT_TEST:
        break;
      case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        _display_handle_display_change(display);
        break;
      case SDL_EVENT_WINDOW_SAFE_AREA_CHANGED:
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
      case SDL_EVENT_CLIPBOARD_UPDATE:
        break;
      case SDL_EVENT_DROP_FILE:
        _display_handle_drop_event(file_drop_array, event.drop);
        break;
      case SDL_EVENT_DROP_TEXT:
      case SDL_EVENT_DROP_BEGIN:
      case SDL_EVENT_DROP_COMPLETE:
      case SDL_EVENT_DROP_POSITION:
        break;
      default:
        warn_message("Unhandled SDL event type: %u.", event.type);
        break;
    }
  }
}

void display_handle_maximize(Display* display, bool reinstate_position) {
  MD_CHECK_NULL_ARGUMENT(display);

  uint32_t width;
  uint32_t height;
  int x, y;

  if (display->is_maximized) {
    width  = display->unmaximize_width;
    height = display->unmaximize_height;

    x = display->unmaximize_x;
    y = display->unmaximize_y;
  }
  else {
    const SDL_Rect bounds = display->screen_bounds;
    int window_width, window_height;
    if (
      bounds.w <= 0 || bounds.h <= 0 || SDL_GetWindowSize(display->sdl_window, &window_width, &window_height) == false
      || SDL_GetWindowPosition(display->sdl_window, &display->unmaximize_x, &display->unmaximize_y) == false)
      return;

    display->unmaximize_width  = (uint32_t) window_width;
    display->unmaximize_height = (uint32_t) window_height;
    width                      = (uint32_t) bounds.w;
    height                     = (uint32_t) bounds.h;
    x                          = bounds.x;
    y                          = bounds.y;
  }

  display->is_maximized = display->is_maximized == false;

  display_resize(display, width, height);

  if (reinstate_position)
    SDL_SetWindowPosition(display->sdl_window, x, y);
}

void display_resize(Display* display, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(display);

  const uint32_t max_width  = (display->screen_bounds.w > 0) ? (uint32_t) display->screen_bounds.w : INT_MAX;
  const uint32_t max_height = (display->screen_bounds.h > 0) ? (uint32_t) display->screen_bounds.h : INT_MAX;
  SDL_SetWindowSize(display->sdl_window, (int) max(1u, min(max_width, width)), (int) max(1u, min(max_height, height)));
}

void display_destroy(Display** display) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(*display);

  display_screenshot_destroy(*display);

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
  render_region_destroy(&(*display)->region);
  display_zoom_handler_destroy(&(*display)->zoom_handler);
  file_dialog_handler_destroy(&(*display)->scene_file_path);
  file_dialog_handler_destroy(&(*display)->entity_properties_path);

  LUM_FAILURE_HANDLE(array_destroy(&(*display)->status_messages));

  LUM_FAILURE_HANDLE(host_free(display));

  __num_displays--;

  if (__num_displays == 0) {
    SDL_Quit();
    TTF_Quit();
  }
}

uint32_t display_add_status_message(Display* display, const char* name, const char* string) {
  MD_CHECK_NULL_ARGUMENT(display);

  DisplayStatusMessage message;
  memset(&message, 0, sizeof(DisplayStatusMessage));

  strncpy(message.name, name ? name : "", sizeof(message.name) - 1);
  strncpy(message.string, string ? string : "", sizeof(message.string) - 1);
  message.start_time_ns = SDL_GetTicksNS();
  message.active        = true;
  message.id            = 0;

  uint32_t num_elements;
  array_get_num_elements(display->status_messages, &num_elements);

  for (; message.id < num_elements; message.id++) {
    if (display->status_messages[message.id].active == false) {
      break;
    }
  }

  if (message.id == num_elements) {
    array_push(&display->status_messages, &message);
  }
  else {
    display->status_messages[message.id] = message;
  }

  return message.id;
}

void display_remove_status_message(Display* display, uint32_t id) {
  MD_CHECK_NULL_ARGUMENT(display);

  if (id == (uint32_t) -1)
    return;

  uint32_t num_elements;
  array_get_num_elements(display->status_messages, &num_elements);

  if (id >= num_elements)
    crash_message("Message is out of bounds.");

  display->status_messages[id].active = false;
}
