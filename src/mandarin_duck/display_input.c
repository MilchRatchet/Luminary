#include <float.h>

#include "display_internal.h"

void display_set_cursor(Display* display, SDL_SystemCursor cursor) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->selected_cursor = cursor;
}

static void _display_move_sun(Display* display, LuminaryHost* host, float time_step) {
  const KeyboardState* keyboard = display->keyboard_state;
  const int azimuth_direction   = keyboard->keys[SDL_SCANCODE_RIGHT].down - keyboard->keys[SDL_SCANCODE_LEFT].down;
  const int altitude_direction  = keyboard->keys[SDL_SCANCODE_UP].down - keyboard->keys[SDL_SCANCODE_DOWN].down;

  if ((azimuth_direction == 0 && altitude_direction == 0) || time_step == 0.0f)
    return;

  LuminarySky sky;
  LUM_FAILURE_HANDLE(luminary_host_get_sky(host, &sky));

  sky.azimuth += azimuth_direction * (0.2f * time_step);
  sky.altitude += altitude_direction * (0.2f * time_step);

  LUM_FAILURE_HANDLE(luminary_host_set_sky(host, &sky));
}

static void _display_set_default_cursor(Display* display) {
  SDL_SystemCursor cursor = SDL_SYSTEM_CURSOR_DEFAULT;

  if (display->mouse_mode == DISPLAY_MOUSE_MODE_SELECT || display->mouse_mode == DISPLAY_MOUSE_MODE_FOCUS) {
    bool mouse_hovers_background = false;
    user_interface_mouse_hovers_background(display->ui, display, &mouse_hovers_background);
    if (mouse_hovers_background)
      cursor = SDL_SYSTEM_CURSOR_POINTER;
  }

  display_set_cursor(display, cursor);
}

static LuminaryPixelQueryResult* _display_get_pixel_query_result(Display* display) {
  switch (display->mouse_mode) {
    case DISPLAY_MOUSE_MODE_DEFAULT:
      return &display->move_pixel_data;
    case DISPLAY_MOUSE_MODE_SELECT:
      return &display->select_pixel_data;
    case DISPLAY_MOUSE_MODE_FOCUS:
      return &display->focus_pixel_data;
    default:
      return NULL;
  }
}

static void _display_query_pixel_info(Display* display, LuminaryHost* host, float request_x, float request_y) {
  LuminaryPixelQueryResult* target = _display_get_pixel_query_result(display);

  const uint32_t screen_x = (uint32_t) fmaxf(request_x, 0.0f);
  const uint32_t screen_y = (uint32_t) fmaxf(request_y, 0.0f);

  uint32_t image_x, image_y;
  display_zoom_handler_screen_to_image(display->zoom_handler, screen_x, screen_y, &image_x, &image_y);

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  if (image_x >= settings.width || image_y >= settings.height) {
    display->awaiting_pixel_query_result = false;

    if (target != NULL)
      target->pixel_query_is_valid = false;

    return;
  }

  const uint16_t x = (uint16_t) image_x;
  const uint16_t y = (uint16_t) image_y;

  LuminaryPixelQueryResult result;

  LUM_FAILURE_HANDLE(luminary_host_get_pixel_info(host, x, y, &result));

  display->reference_x = screen_x;
  display->reference_y = screen_y;

  display->awaiting_pixel_query_result = result.pixel_query_is_valid == false;

  if (target != NULL)
    *target = result;
}

static void _display_start_camera_mode(Display* display, const bool left_pressed) {
  display->active_camera_movement = true;
  display_set_mouse_visible(display, false);

  camera_handler_set_mode(display->camera_handler, (left_pressed) ? CAMERA_MODE_ORBIT : CAMERA_MODE_ZOOM);
  camera_handler_set_reference_pos(display->camera_handler, display->move_pixel_data.rel_hit_pos);
}

WindowVisibilityMask display_get_window_visibility_mask(Display* display) {
  if (display->show_ui) {
    return (display->active_camera_movement) ? WINDOW_VISIBILITY_MASK_MOVEMENT : WINDOW_VISIBILITY_MASK_ALL;
  }

  return WINDOW_VISIBILITY_NONE;
}

static void _display_sync_resolution(Display* display, LuminaryHost* host) {
  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  if (display->zoom_handler->image_width != settings.width || display->zoom_handler->image_height != settings.height) {
    display_zoom_handler_set_image_size(display->zoom_handler, settings.width, settings.height);
  }

  if (display->region->image_width != settings.width || display->region->image_height != settings.height) {
    render_region_handler_set_image_size(display->region, settings.width, settings.height);
  }

  LuminaryOutputProperties output_properties;
  LUM_FAILURE_HANDLE(luminary_host_get_output_properties(host, &output_properties));

  if (output_properties.width != settings.width || output_properties.height != settings.height) {
    output_properties.width  = settings.width;
    output_properties.height = settings.height;

    LUM_FAILURE_HANDLE(luminary_host_set_output_properties(host, output_properties));
  }
}

static void _display_handle_global_actions(Display* display, LuminaryHost* host) {
  if (display->keyboard_state->keys[SDL_SCANCODE_E].phase == KEY_PHASE_RELEASED) {
    display->show_ui                = display->show_ui == false;
    display->active_camera_movement = display->show_ui == false;
    display_set_mouse_visible(display, display->show_ui);
  }

  if (display->keyboard_state->keys[SDL_SCANCODE_F2].phase == KEY_PHASE_PRESSED) {
    display_screenshot_request(display, host);
  }

  if (display->show_ui && display->keyboard_state->keys[SDL_SCANCODE_S].phase == KEY_PHASE_PRESSED) {
    const bool ctrl_down = display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down;

    if (ctrl_down) {
      display_scene_request_save(display);
    }
  }

  display_scene_handle_save(display, host);
}

static void _display_retry_pixel_query(Display* display, LuminaryHost* host) {
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
}

static void _display_handle_viewport_actions(Display* display, LuminaryHost* host, UserInterfaceStatus ui_status) {
  if (display->show_ui) {
    if (ui_status.received_keyboard_action == false) {
      if (display->keyboard_state->keys[SDL_SCANCODE_1].phase == KEY_PHASE_PRESSED) {
        display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_DEFAULT);
      }

      if (display->keyboard_state->keys[SDL_SCANCODE_2].phase == KEY_PHASE_PRESSED) {
        display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_SELECT);
      }

      if (display->keyboard_state->keys[SDL_SCANCODE_3].phase == KEY_PHASE_PRESSED) {
        display_set_mouse_mode(display, DISPLAY_MOUSE_MODE_FOCUS);
      }
    }

    if (
      (ui_status.received_hover == false) && (ui_status.received_mouse_action == false) && (ui_status.received_keyboard_action == false)) {
      switch (display->mouse_mode) {
        case DISPLAY_MOUSE_MODE_DEFAULT:
        default:
          if (display->mouse_visible) {
            const bool left_pressed  = display->mouse_state->phase == MOUSE_PHASE_PRESSED;
            const bool right_pressed = display->mouse_state->right_phase == MOUSE_PHASE_PRESSED;
            const bool alt_down      = display->keyboard_state->keys[SDL_SCANCODE_LALT].down;
            const bool ctrl_down     = display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down;

            if ((left_pressed || right_pressed) && alt_down && (ctrl_down == false)) {
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
            const bool ctrl_down     = display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down;

            const bool orbit_released = display->camera_handler->mode == CAMERA_MODE_ORBIT && left_pressed == false;
            const bool zoom_released  = display->camera_handler->mode == CAMERA_MODE_ZOOM && right_pressed == false;

            if (orbit_released || zoom_released || alt_down == false || ctrl_down) {
              display->active_camera_movement = false;
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

        camera.lens.object_distance      = fmaxf(display->focus_pixel_data.depth, 0.01f);
        camera.thin_lens.object_distance = fmaxf(display->focus_pixel_data.depth, 0.01f);

        LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));

        // Invalidate the pixel query so that it does not keep overriding the focal length now.
        display->focus_pixel_data.pixel_query_is_valid = false;
      }
    }
  }
}

void display_handle_inputs(Display* display, LuminaryHost* host, float time_step) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  _display_sync_resolution(display, host);
  display->frametime = time_step;

  _display_handle_global_actions(display, host);
  _display_retry_pixel_query(display, host);
  _display_move_sun(display, host, time_step);
  _display_set_default_cursor(display);

  WindowVisibilityMask visibility_mask = display_get_window_visibility_mask(display);
  UserInterfaceStatus ui_status        = user_interface_handle_inputs(display->ui, display, host, visibility_mask);
  _display_handle_viewport_actions(display, host, ui_status);

  if (display->active_camera_movement) {
    camera_handler_update(display->camera_handler, host, display->keyboard_state, display->mouse_state, time_step);
  }
  else if (ui_status.received_keyboard_action == false) {
    const bool ctrl_down = display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down || display->keyboard_state->keys[SDL_SCANCODE_RCTRL].down;
    display_zoom_handler_update(display->zoom_handler, display->mouse_state, ctrl_down);
    render_region_handle_inputs(display->region, display, host, display->mouse_state, display->keyboard_state);
  }
}
