#include "user_interface.h"

#include "display.h"
#include "windows/caption_controls.h"
#include "windows/entity_properties.h"
#include "windows/frametime.h"
#include "windows/renderer_status.h"
#include "windows/sidebar.h"

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

static void _user_interface_setup(UserInterface* ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  Window* window_entity_properties_camera;
  window_entity_properties_create(&window_entity_properties_camera);

  uint32_t entity_properties_camera_id;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &entity_properties_camera_id));

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_entity_properties_camera));

  Window* window_caption_controls;
  window_caption_controls_create(&window_caption_controls);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_caption_controls));

  Window* window_frametime;
  window_frametime_create(&window_frametime);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_frametime));

  Window* window_renderer_status;
  window_renderer_status_create(&window_renderer_status);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_renderer_status));

  Window* window_sidebar_entities;
  window_sidebar_create(&window_sidebar_entities);
  window_sidebar_register_window_id(window_sidebar_entities, WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA, entity_properties_camera_id);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_sidebar_entities));
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void user_interface_create(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  LUM_FAILURE_HANDLE(host_malloc(ui, sizeof(UserInterface)));

  LUM_FAILURE_HANDLE(array_create(&(*ui)->windows, sizeof(Window*), 16));

  _user_interface_setup(*ui);
}

void user_interface_mouse_hovers_background(UserInterface* ui, Display* display, bool* mouse_hovers_background) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(mouse_hovers_background);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  bool window_handled_mouse = false;

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    if (window->is_visible == false)
      continue;

    window_handled_mouse |= window_is_mouse_hover(window, display);
  }

  *mouse_hovers_background = !window_handled_mouse;
}

void user_interface_handle_inputs(UserInterface* ui, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  bool window_handled_mouse = false;

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    if (window->is_visible == false)
      continue;

    window_handled_mouse |= window_handle_input(window, display, host);
  }
}

void user_interface_render(UserInterface* ui, Display* display) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    if (window->is_visible == false)
      continue;

    window_render(window, display);
  }

  ui_renderer_render_display_corners(display->ui_renderer, display);
}

void user_interface_destroy(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(*ui);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements((*ui)->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    window_destroy(&(*ui)->windows[window_id]);
  }

  LUM_FAILURE_HANDLE(array_destroy(&(*ui)->windows));

  LUM_FAILURE_HANDLE(host_free(ui));
}

void user_interface_get_window_visible(UserInterface* ui, uint32_t window_id, bool* visible) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(visible);

  *visible = ui->windows[window_id]->is_visible;
}

void user_interface_set_window_visible(UserInterface* ui, uint32_t window_id, bool visible) {
  MD_CHECK_NULL_ARGUMENT(ui);

  ui->windows[window_id]->is_visible = visible;
}
